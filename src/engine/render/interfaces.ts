export interface RenderPass {
    allocUniform(): {
        index: number;
        offset: number;
        buffer: GPUBuffer;
        model: Float32Array;
        layout: GPUBindGroupLayout;
    };

    freeUniformIndex(index: number): void;
}

export interface Renderable {
    draw(pass: GPURenderPassEncoder);
    free();
}
