# DotStories Web Engine

Run large AI models in the browser with WebGPU. No install, no server: the model runs on your own GPU.

> **Status: experimental.** We are building a proof of concept with [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo)
> (12B image model). Expect breaking changes.

Part of [DotStories](https://dotstories.com), an open-source community for sharing AI workflows.

## How it works

- Model weights are streamed from browser storage straight into GPU memory, one layer at a time.
- Quantized weights (int4/fp8) are unpacked inside the GPU shaders.
- Every kernel is tested against a Python reference.

## Requirements

- A desktop browser with WebGPU (Chrome or Edge 113+).
- A GPU with `shader-f16` support and enough memory (see the probe page).

## Run it locally

```bash
pnpm install
pnpm dev
```

## Tests

```bash
pnpm lint && pnpm typecheck && pnpm test
pnpm test:e2e
```

## Contributing

Issues and pull requests are welcome. See the [DotStories contributing guide](https://github.com/dotstories/dotstories/blob/main/CONTRIBUTING.md)
and [code of conduct](https://github.com/dotstories/dotstories/blob/main/CODE_OF_CONDUCT.md).
Contributions are licensed under the Apache License 2.0.

Security issues: please report them privately, see [SECURITY.md](https://github.com/dotstories/dotstories/blob/main/SECURITY.md).

## License

Code: [Apache-2.0](LICENSE), see [NOTICE](NOTICE). Models keep their own licenses: you must accept a model's license before you use it.
