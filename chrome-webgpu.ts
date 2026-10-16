// CI has no GPU: Chrome's software WebGPU (SwiftShader) runs the same shaders, only slower.
// On a computer with a GPU, Chrome picks the real one when these flags are left out.
export const webgpuFlags = ["--enable-unsafe-webgpu", "--enable-unsafe-swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"];
