import type { Graph } from "../gpu/graph";
import { toFloat32 } from "../weights/fp8";
import { readTensor, type Safetensors } from "../weights/safetensors";
import type { AttentionMlpWeights, BlockWeights } from "./block";
import type { Linear } from "./ops";
import type { TransformerWeights } from "./transformer";

/** Loads tensors from the person's model file straight into GPU buffers. Names are ComfyUI's. */

function tensor(st: Safetensors, name: string) {
  const t = st.tensors.get(name);
  if (!t) throw new Error(`Missing tensor ${name}: is this a Krea 2 file?`);
  return t;
}

/** A linear layer as stored: fp8 [n, k] with its scalar weight_scale, or bf16 [n, k]; plus its bias if it has one. */
export async function loadLinear(graph: Graph, file: Blob, st: Safetensors, name: string): Promise<Linear> {
  const w = tensor(st, `${name}.weight`);
  if (w.shape.length !== 2 || (w.dtype !== "F8_E4M3" && w.dtype !== "BF16")) {
    throw new Error(`${name}.weight: expected a 2D F8_E4M3 or BF16 tensor, got ${w.dtype} ${JSON.stringify(w.shape)}`);
  }
  const bytes = await readTensor(file, w);
  const fp8 = w.dtype === "F8_E4M3";
  const bias = st.tensors.has(`${name}.bias`) ? graph.buffer(await loadFloats(file, st, `${name}.bias`)) : undefined;
  return {
    buffer: graph.buffer(new Uint32Array(bytes.buffer as ArrayBuffer, bytes.byteOffset, bytes.byteLength / 4)),
    dtype: fp8 ? "fp8" : "bf16",
    scale: fp8 ? (await loadFloats(file, st, `${name}.weight_scale`))[0] : 1,
    n: w.shape[0],
    k: w.shape[1],
    bias,
  };
}

/** Any float tensor as f32 on the CPU (norm weights, modulation tables are small). */
export async function loadFloats(file: Blob, st: Safetensors, name: string): Promise<Float32Array<ArrayBuffer>> {
  const t = tensor(st, name);
  return toFloat32(t.dtype, await readTensor(file, t));
}

/** Attention + MLP + norms: the same names in single-stream and text fusion blocks. */
async function loadAttentionMlp(graph: Graph, file: Blob, st: Safetensors, prefix: string): Promise<AttentionMlpWeights> {
  const lin = (name: string) => loadLinear(graph, file, st, `${prefix}.${name}`);
  const vec = async (name: string) => graph.buffer(await loadFloats(file, st, `${prefix}.${name}`));
  return {
    wq: await lin("attn.wq"),
    wk: await lin("attn.wk"),
    wv: await lin("attn.wv"),
    gate: await lin("attn.gate"),
    wo: await lin("attn.wo"),
    mlpGate: await lin("mlp.gate"),
    mlpUp: await lin("mlp.up"),
    mlpDown: await lin("mlp.down"),
    norm1: await vec("prenorm.scale"),
    norm2: await vec("postnorm.scale"),
    normQ: await vec("attn.qknorm.qnorm.scale"),
    normK: await vec("attn.qknorm.knorm.scale"),
  };
}

export async function loadBlock(graph: Graph, file: Blob, st: Safetensors, index: number): Promise<BlockWeights> {
  const prefix = `blocks.${index}`;
  return { ...(await loadAttentionMlp(graph, file, st, prefix)), table: graph.buffer(await loadFloats(file, st, `${prefix}.mod.lin`)) };
}

/** Everything the transformer needs, with only the first `blocks` blocks (28 for the full model). */
export async function loadTransformer(
  graph: Graph,
  file: Blob,
  st: Safetensors,
  blocks: number,
  onProgress?: (done: number, total: number) => void,
): Promise<TransformerWeights> {
  const lin = (name: string) => loadLinear(graph, file, st, name);
  const vec = async (name: string) => graph.buffer(await loadFloats(file, st, name));
  const fusion = (kind: string) => Promise.all([0, 1].map((i) => loadAttentionMlp(graph, file, st, `txtfusion.${kind}.${i}`)));
  const lastTable = await loadFloats(file, st, "last.modulation.lin"); // [2, hidden]: scale, shift
  const hidden = lastTable.length / 2;

  const w: Omit<TransformerWeights, "blocks"> = {
    first: await lin("first"),
    tmlp0: await lin("tmlp.0"),
    tmlp2: await lin("tmlp.2"),
    tproj: await lin("tproj.1"),
    layerwise: await fusion("layerwise_blocks"),
    projector: await vec("txtfusion.projector.weight"),
    refiner: await fusion("refiner_blocks"),
    txtNorm: await vec("txtmlp.0.scale"),
    txt1: await lin("txtmlp.1"),
    txt3: await lin("txtmlp.3"),
    lastNorm: await vec("last.norm.scale"),
    lastScale: graph.buffer(lastTable.subarray(0, hidden)),
    lastShift: graph.buffer(lastTable.subarray(hidden)),
    lastLinear: await lin("last.linear"),
  };
  const loaded: BlockWeights[] = [];
  for (let i = 0; i < blocks; i++) {
    loaded.push(await loadBlock(graph, file, st, i));
    onProgress?.(i + 1, blocks);
    await graph.device.queue.onSubmittedWorkDone(); // let uploads finish before reading more (bounded RAM)
  }
  return { ...w, blocks: loaded };
}
