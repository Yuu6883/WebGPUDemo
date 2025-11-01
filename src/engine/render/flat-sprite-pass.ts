import Renderer, { checkDevice, GDevice } from './base';

import SpriteVertWGSL from '../shaders/sprite.flat.vert.wgsl'; // reuse your vertex shader
import SpriteFragWGSL from '../shaders/sprite.flat.frag.wgsl'; // new forward fragment shader (per-material lighting)
import { RenderPass } from './interfaces';
import SpriteStore from './sprite-store';

const CHUNK_DIM = 32;

export class RenderChunk {
    public readonly index: number;
    private readonly pass: FlatSpritePass;
    public group: GPUBindGroup = null;
    public elem = 0;

    constructor(pass: FlatSpritePass, index: number) {
        this.pass = pass;
        this.index = index;
    }

    free() {
        this.pass.freeIndices.push(this.index);
        this.elem = 0;
        this.group = null;
    }
}

export class FlatSpritePass implements RenderPass {
    private readonly store: SpriteStore;

    private basePipeline: GPURenderPipeline;
    private vertModule: GPUShaderModule;
    private fragModule: GPUShaderModule;

    private posXBuffer: GPUBuffer;
    private posYBuffer: GPUBuffer;
    private stateBuffer: GPUBuffer;

    private uniformGroup: GPUBindGroup;

    private chunkCapacity = 1024;
    private readonly chunks: RenderChunk[] = [];
    public readonly freeIndices: number[] = [];

    constructor(store: SpriteStore) {
        this.store = store;
    }

    allocChunk() {
        let index = this.freeIndices.shift();
        let chunk: RenderChunk;

        if (index === undefined) {
            index = this.chunks.length;

            chunk = new RenderChunk(this, index);
            this.chunks.push(chunk);
        } else {
            chunk = this.chunks[index];
        }

        if (this.chunks.length > this.chunkCapacity) {
            this.chunkCapacity += 1024;
            this.expandBuffers();
        }
        return chunk;
    }

    updateBindgroup(chunk: RenderChunk) {
        if (!this.posXBuffer || !this.posYBuffer || !this.stateBuffer) return;

        const device = GDevice.device;
        const offset = chunk.index * CHUNK_DIM * CHUNK_DIM * Int32Array.BYTES_PER_ELEMENT;
        chunk.group = device.createBindGroup({
            layout: this.basePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.posXBuffer,
                        offset,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.posYBuffer,
                        offset,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.stateBuffer,
                        offset,
                    },
                },
            ],
        });
    }

    async init() {
        checkDevice();
        if (this.basePipeline) return;
        const device = GDevice.device;

        this.expandBuffers();
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
                targets: [
                    {
                        format: GDevice.format,
                        blend: {
                            color: {
                                srcFactor: 'src-alpha',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                            alpha: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                        },
                    },
                ],
            },
            depthStencil: {
                depthWriteEnabled: false,
                depthCompare: 'less',
                format: 'depth24plus-stencil8',
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',
            },
        });

        this.uniformGroup = device.createBindGroup({
            label: 'Sprite Sampler+Texture+Uniforms',
            layout: this.basePipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: this.store.spriteTexture.createView(),
                },
                {
                    binding: 1,
                    resource: this.store.spriteSampler,
                },
                { binding: 3, resource: { buffer: this.store.spriteParamsBuffer } },
                { binding: 4, resource: { buffer: this.store.camUB } },
            ],
        });
    }

    expandBuffers() {
        const device = GDevice.device;
        const encoder = device.createCommandEncoder();

        const bufferSize =
            this.chunkCapacity * CHUNK_DIM * CHUNK_DIM * Int32Array.BYTES_PER_ELEMENT;
        if (bufferSize <= this.posXBuffer?.size) return;
        console.log(
            `Expanding chunk buffers size to ${(bufferSize / 1024 / 1024).toFixed(3)}MB`,
        );

        const oldX = this.posXBuffer;
        this.posXBuffer = device.createBuffer({
            label: 'Chunk Sprite X Positions',
            size: bufferSize,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST |
                GPUBufferUsage.COPY_SRC,
        });
        if (oldX) encoder.copyBufferToBuffer(oldX, 0, this.posXBuffer, 0, oldX.size);

        const oldY = this.posYBuffer;
        this.posYBuffer = device.createBuffer({
            label: 'Sprite Y Positions',
            size: bufferSize,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST |
                GPUBufferUsage.COPY_SRC,
        });
        if (oldY) encoder.copyBufferToBuffer(oldY, 0, this.posYBuffer, 0, oldY.size);

        const oldState = this.stateBuffer;
        this.stateBuffer = device.createBuffer({
            label: 'Sprite States',
            size: bufferSize,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST |
                GPUBufferUsage.COPY_SRC,
        });
        if (oldState)
            encoder.copyBufferToBuffer(oldState, 0, this.stateBuffer, 0, oldState.size);

        const cmd = encoder.finish();
        device.queue.submit([cmd]);
        device.queue.onSubmittedWorkDone().then(() => {
            oldX?.destroy();
            oldY?.destroy();
            oldState?.destroy();
        });

        for (const chunk of this.chunks) chunk.group = null;
    }

    updateChunk(
        chunk: RenderChunk,
        posX: Int32Array,
        posY: Int32Array,
        state: Uint32Array,
    ) {
        const device = GDevice.device;
        const queue = device.queue;

        if (posX.length !== posY.length || posX.length !== state.length) {
            throw new Error('Sprite buffer lengths mismatch');
        }
        if (
            posX.length > CHUNK_DIM * CHUNK_DIM ||
            posY.length > CHUNK_DIM * CHUNK_DIM ||
            state.length > CHUNK_DIM * CHUNK_DIM
        ) {
            throw new Error('Chunk buffer lengths overflow');
        }

        if (!this.posXBuffer || !this.posYBuffer || !this.stateBuffer) return;

        const offset = chunk.index * CHUNK_DIM * CHUNK_DIM * Int32Array.BYTES_PER_ELEMENT;
        queue.writeBuffer(this.posXBuffer, offset, posX.buffer);
        queue.writeBuffer(this.posYBuffer, offset, posY.buffer);
        queue.writeBuffer(this.stateBuffer, offset, state.buffer);
        chunk.elem = posX.length;
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

    render(dt: number, now: number, output: GPUTexture, clear: boolean) {
        if (!this.basePipeline) return;

        const device = GDevice.device;
        const queue = device.queue;

        // single-pass forward render to provided output (color + depth)
        const cmd = device.createCommandEncoder();

        const colorAttachment: GPURenderPassColorAttachment = {
            view: output.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: clear ? 'clear' : 'load',
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
        pass.setBindGroup(1, this.uniformGroup);

        for (const chunk of this.chunks) {
            if (!chunk.group) this.updateBindgroup(chunk);
            if (!chunk.elem) continue;
            pass.setBindGroup(0, chunk.group);
            pass.draw(6, chunk.elem);
        }

        pass.end();
        queue.submit([cmd.finish()]);
    }
}
