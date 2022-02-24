export interface Renderable {
    draw(pass: GPURenderPassEncoder);
    free();
}
