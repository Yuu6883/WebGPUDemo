export interface RenderPass {
    init();
    resize(width: number, height: number);
    allocUniform(): {
        index: number;
        offset: number;
        buffer: GPUBuffer;
        model: Float32Array;
        layout: GPUBindGroupLayout;
    };

    freeUniformIndex(index: number): void;
    render(dt: number, now: number, output: GPUTexture, clear: boolean);
}

export interface Renderable {
    draw(pass: GPURenderPassEncoder);
    free();
}
