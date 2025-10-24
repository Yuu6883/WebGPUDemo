import Renderer, { checkDevice, GDevice } from './base';

import SpriteVertWGSL from '../shaders/sprite.vert.wgsl'; // reuse your vertex shader
import SpriteFragWGSL from '../shaders/unlit.frag.wgsl'; // new forward fragment shader (per-material lighting)
import { RenderPass } from './interfaces';

export class SpritePass implements RenderPass {
    private camUB: GPUBuffer;

    private basePipeline: GPURenderPipeline;
    private vertModule: GPUShaderModule;
    private fragModule: GPUShaderModule;
    private viewGroup: GPUBindGroup;

    readonly spriteParams = new Float32Array(65536 * 4);

    private drawNum = 0;
    private posXBuffer: GPUBuffer;
    private posYBuffer: GPUBuffer;
    private stateBuffer: GPUBuffer;
    private spriteParamsBuffer: GPUBuffer;
    private tickBuffer: GPUBuffer;

    async init() {
        checkDevice();
        if (this.basePipeline) return;
        const device = GDevice.device;

        this.camUB = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT * 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // create shaders
        this.vertModule = device.createShaderModule({ code: SpriteVertWGSL });
        this.fragModule = device.createShaderModule({ code: SpriteFragWGSL });

        // pipeline
        this.basePipeline = await device.createRenderPipelineAsync({
            layout: 'auto',
            vertex: {
                module: this.vertModule,
                entryPoint: 'main',
                buffers: [],
            },
            fragment: {
                module: this.fragModule,
                entryPoint: 'main',
                targets: [{ format: GDevice.format }],
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus-stencil8',
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
        });

        this.spriteParamsBuffer = device.createBuffer({
            label: 'Sprite Params',
            size: Float32Array.BYTES_PER_ELEMENT * 65536 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.tickBuffer = device.createBuffer({
            size: BigUint64Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.expandBuffers();

        for (let i = 0; i < 5; i++) {
            const o = i * 4;
            this.spriteParams[o + 0] = 0.75 / 2;
            this.spriteParams[o + 1] = 0.75 / 2;
        }

        const queue = device.queue;
        queue.writeBuffer(this.spriteParamsBuffer, 0, this.spriteParams);
    }

    expandBuffers() {
        if (!this.basePipeline) return;
        const device = GDevice.device;

        const bufferSize =
            (Math.floor(this.drawNum / 4096) + 1) * 4096 * Int32Array.BYTES_PER_ELEMENT; // 4 bytes per sprite for i32/u32

        if (bufferSize <= this.posXBuffer?.size) return;
        console.log(
            `Expanding buffer size to ${(bufferSize / 1024 / 1024).toFixed(3)}MB`,
        );

        this.posXBuffer?.destroy();
        this.posYBuffer?.destroy();
        this.stateBuffer?.destroy();

        this.posXBuffer = device.createBuffer({
            label: 'Sprite X Positions',
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.posYBuffer = device.createBuffer({
            label: 'Sprite Y Positions',
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.stateBuffer = device.createBuffer({
            label: 'Sprite States',
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // bind group
        this.viewGroup = device.createBindGroup({
            layout: this.basePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.posXBuffer } },
                { binding: 1, resource: { buffer: this.posYBuffer } },
                { binding: 2, resource: { buffer: this.stateBuffer } },
                { binding: 3, resource: { buffer: this.spriteParamsBuffer } },
                { binding: 4, resource: { buffer: this.camUB } },
                // { binding: 5, resource: { buffer: this.tickBuffer } },
            ],
        });
    }

    updateSprites(posX: Int32Array, posY: Int32Array, state: Uint32Array) {
        const device = GDevice.device;
        const queue = device.queue;

        if (posX.length !== posY.length || posX.length !== state.length) {
            throw new Error('Sprite buffer lengths mismatch');
        }
        const size = (this.drawNum = state.length);
        this.expandBuffers();

        if (!this.posXBuffer || !this.posYBuffer || !this.stateBuffer) return;
        queue.writeBuffer(
            this.posXBuffer,
            0,
            posX.buffer,
            posX.byteOffset,
            Int32Array.BYTES_PER_ELEMENT * size,
        );
        queue.writeBuffer(
            this.posYBuffer,
            0,
            posY.buffer,
            posY.byteOffset,
            Int32Array.BYTES_PER_ELEMENT * size,
        );

        queue.writeBuffer(
            this.stateBuffer,
            0,
            state.buffer,
            state.byteOffset,
            Uint32Array.BYTES_PER_ELEMENT * size,
        );
    }

    resize(width: number, height: number) {}

    allocUniform(): {
        index: number;
        offset: number;
        buffer: GPUBuffer;
        model: Float32Array;
        layout: GPUBindGroupLayout;
    } {
        return null;
    }
    freeUniformIndex() {}

    render(dt: number, now: number, output: GPUTexture, camBuf: Float32Array) {
        if (!this.basePipeline) return;

        const device = GDevice.device;
        const queue = device.queue;

        queue.writeBuffer(
            this.camUB,
            0,
            camBuf.buffer,
            0,
            Float32Array.BYTES_PER_ELEMENT * 16,
        );

        // single-pass forward render to provided output (color + depth)
        const cmd = device.createCommandEncoder();

        const colorAttachment: GPURenderPassColorAttachment = {
            view: output.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
        };

        const passDesc: GPURenderPassDescriptor = {
            colorAttachments: [colorAttachment],
            depthStencilAttachment: {
                view: Renderer.DefaultDepthStencilView,
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
                stencilClearValue: 0,
                stencilLoadOp: 'clear',
                stencilStoreOp: 'store',
            },
        };

        const pass = cmd.beginRenderPass(passDesc);
        pass.setViewport(0, 0, output.width, output.height, 0, 1);
        pass.setPipeline(this.basePipeline);
        pass.setBindGroup(0, this.viewGroup);

        if (this.drawNum) pass.draw(6, this.drawNum);

        pass.end();
        queue.submit([cmd.finish()]);
    }
}
