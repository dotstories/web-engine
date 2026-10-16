import { checkWebGPU } from "./gpu/support";

const MESSAGES = {
  "no-webgpu": "This browser has no WebGPU. Try Chrome or Edge on a desktop.",
  "no-adapter": "WebGPU is on, but no GPU was found.",
};

const status = document.querySelector("#status")!;
const support = await checkWebGPU();
status.textContent = support.ok ? `WebGPU ready: ${support.adapter.info.vendor || "unknown GPU"}` : MESSAGES[support.reason];
