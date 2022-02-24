import { ReadonlyVec3 } from 'gl-matrix';
import Transform from '../core/transform';
import { checkDevice, GDevice } from '../render/base';
import { Renderable } from '../render/pass';
import { DeferredPipeline, GBuffer } from '../render/pipeline';
import ClothWGSL from '../shaders/cloth.wgsl';

export interface ClothConstants {
    mass: number;
    rest_length: number;
    springConstant: number;
    dampingConstant: number;
    gravity: ReadonlyVec3;
    wind: ReadonlyVec3;
}

export const GCloth: {
    shader: GPUShaderModule;
    renderPipeline: GPURenderPipeline;
    forceCalcPipeline: GPUComputePipeline;
    triangleGenPipeline: GPUComputePipeline;
    updatePipeline: GPUComputePipeline;
    dtGroup: GPUBindGroup;
    viewGroup: GPUBindGroup;
} = {
    shader: null,
    renderPipeline: null,
    forceCalcPipeline: null,
    triangleGenPipeline: null,
    updatePipeline: null,
    dtGroup: null,
    viewGroup: null,
};

export default class Cloth implements Renderable {
    // vec3<f32> x4 NOTE: vec3 is padded to 16 bytes
    static readonly STRIDE = Float32Array.BYTES_PER_ELEMENT * 4 * 4;

    private readonly transform: Transform;

    // s_ for simulation
    private readonly s_computeGroup: GPUBindGroup;
    private readonly s_uniformGroup: GPUBindGroup;

    // u_ for update
    private readonly u_computeGroup: GPUBindGroup;
    private readonly u_uniformGroup: GPUBindGroup;

    private readonly pointBuffer: GPUBuffer;
    private readonly constUB: GPUBuffer;
    private readonly dimUB: GPUBuffer;
    private readonly vectorUB: GPUBuffer;

    private readonly constants = new Float32Array(4);
    private readonly dimension = new Uint32Array(2);
    private readonly vectors = new Float32Array(8);

    private readonly ibo: GPUBuffer;

    public readonly uniformIndex: number;
    private readonly modelGroup: GPUBindGroup;
    private readonly pipeline: DeferredPipeline;

    private readonly totalIndices: number;

    public static readonly VertexLayout: GPUVertexBufferLayout = {
        arrayStride: Cloth.STRIDE,
        attributes: [
            {
                // position
                shaderLocation: 0,
                offset: 0,
                format: 'float32x3',
            },
            {
                // normal
                shaderLocation: 1,
                offset: Float32Array.BYTES_PER_ELEMENT * 4, // Padded
                format: 'float32x3',
            },
            {
                // uv ((unused))
                shaderLocation: 2,
                offset: Float32Array.BYTES_PER_ELEMENT * 8,
                format: 'float32x2',
            },
        ],
    };

    static initPipeline(camUB: GPUBuffer, dtUB: GPUBuffer) {
        checkDevice();
        const device = GDevice.device;

        GCloth.shader = device.createShaderModule({
            code: ClothWGSL,
        });

        GCloth.forceCalcPipeline = device.createComputePipeline({
            compute: {
                module: GCloth.shader,
                entryPoint: 'calc_forces',
            },
        });

        GCloth.renderPipeline = device.createRenderPipeline({
            vertex: {
                module: GBuffer.basePassVertShader,
                entryPoint: 'main',
                buffers: [Cloth.VertexLayout],
            },
            fragment: {
                module: GBuffer.basePassFragShader,
                entryPoint: 'main',
                targets: [
                    // position
                    { format: 'rgba32float' },
                    // normal
                    { format: 'rgba32float' },
                    // albedo
                    { format: 'bgra8unorm' },
                ],
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
        });

        GCloth.triangleGenPipeline = device.createComputePipeline({
            compute: {
                module: GCloth.shader,
                entryPoint: 'init_indices',
            },
        });

        GCloth.updatePipeline = device.createComputePipeline({
            compute: {
                module: GCloth.shader,
                entryPoint: 'update',
            },
        });

        GCloth.viewGroup = device.createBindGroup({
            layout: GCloth.renderPipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: camUB,
                    },
                },
            ],
        });

        GCloth.dtGroup = device.createBindGroup({
            layout: GCloth.updatePipeline.getBindGroupLayout(2),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: dtUB,
                    },
                },
            ],
        });
    }

    constructor(
        pipeline: DeferredPipeline,
        width: number,
        height: number,
        constants: ClothConstants,
    ) {
        checkDevice();
        this.pipeline = pipeline;
        const device = GDevice.device;

        this.dimension.set([width, height]);
        this.constants.set([
            constants.mass,
            constants.rest_length,
            constants.springConstant,
            constants.dampingConstant,
        ]);

        this.vectors.set(constants.wind, 0);
        this.vectors.set(constants.gravity, 4);

        this.pointBuffer = device.createBuffer({
            size: Cloth.STRIDE * this.dimension[0] * this.dimension[1],
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.VERTEX |
                // GPUBufferUsage.COPY_DST |
                GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });

        const RLEN = constants.rest_length;
        const mapped = this.pointBuffer.getMappedRange();
        for (let y = 0; y < this.dimension[0]; y++) {
            for (let x = 0; x < this.dimension[1]; x++) {
                const offset = Cloth.STRIDE * (y * this.dimension[0] + x);
                // Position (set top row to fixed points by setting their position.w to -1)
                new Float32Array(mapped, offset, 4).set([
                    x * RLEN,
                    y * RLEN,
                    0,
                    y === this.dimension[0] - 1 ? -1 : 0,
                ]);
                // Normal
                new Float32Array(mapped, offset + 12, 3).set([0, 0, -1]);
            }
        }
        this.pointBuffer.unmap();

        console.log('Points uploaded to GPU');

        this.constUB = device.createBuffer({
            size: this.constants.byteLength,
            usage: GPUBufferUsage.UNIFORM,
            mappedAtCreation: true,
        });

        // Upload constants buffer to gpu uniform
        new Float32Array(this.constUB.getMappedRange()).set(this.constants);
        this.constUB.unmap();

        this.dimUB = device.createBuffer({
            size: this.dimension.byteLength,
            usage: GPUBufferUsage.UNIFORM,
            mappedAtCreation: true,
        });

        // Upload constants buffer to gpu uniform
        new Uint32Array(this.dimUB.getMappedRange()).set(this.dimension);
        this.dimUB.unmap();

        this.vectorUB = device.createBuffer({
            size: this.vectors.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        // Upload constants buffer to gpu uniform
        new Float32Array(this.vectorUB.getMappedRange()).set(this.vectors);
        this.vectorUB.unmap();

        this.s_computeGroup = device.createBindGroup({
            layout: GCloth.forceCalcPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.pointBuffer },
                },
            ],
        });

        this.s_uniformGroup = device.createBindGroup({
            layout: GCloth.forceCalcPipeline.getBindGroupLayout(1),
            entries: [this.constUB, this.dimUB, this.vectorUB].map((buffer, binding) => ({
                binding,
                resource: { buffer },
            })),
        });

        this.u_computeGroup = device.createBindGroup({
            layout: GCloth.updatePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.pointBuffer },
                },
            ],
        });

        this.u_uniformGroup = device.createBindGroup({
            layout: GCloth.updatePipeline.getBindGroupLayout(1),
            entries: [this.constUB, this.dimUB].map((buffer, binding) => ({
                binding,
                resource: { buffer },
            })),
        });

        // Generate IBO on gpu because why not
        {
            // 2 tri per quad, 3 indices per triangle
            this.totalIndices = (this.dimension[0] - 1) * (this.dimension[1] - 1) * 2 * 3;

            const idxBufSize = Uint32Array.BYTES_PER_ELEMENT * 3 * this.totalIndices;
            this.ibo = device.createBuffer({
                size: idxBufSize,
                usage: GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE,
            });

            console.log(`Total cloth triangles: ${this.totalIndices}`);

            const indicesGroup0 = device.createBindGroup({
                layout: GCloth.triangleGenPipeline.getBindGroupLayout(0),
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.ibo },
                    },
                ],
            });

            const indicesGroup1 = device.createBindGroup({
                layout: GCloth.triangleGenPipeline.getBindGroupLayout(1),
                entries: [
                    {
                        binding: 1,
                        resource: {
                            buffer: this.dimUB,
                        },
                    },
                ],
            });

            const GridX = Math.ceil((this.dimension[0] - 1) / 16);
            const GridY = Math.ceil((this.dimension[1] - 1) / 16);

            const cmd = device.createCommandEncoder();
            const indicesPass = cmd.beginComputePass();
            indicesPass.setPipeline(GCloth.triangleGenPipeline);
            indicesPass.setBindGroup(0, indicesGroup0);
            indicesPass.setBindGroup(1, indicesGroup1);
            indicesPass.dispatch(GridX, GridY);
            indicesPass.endPass();

            // const bufSize = Cloth.STRIDE * this.dimension[0] * this.dimension[1];
            // const testBuf = device.createBuffer({
            //     size: idxBufSize,
            //     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            // });

            // cmd.copyBufferToBuffer(this.ibo, 0, testBuf, 0, idxBufSize);

            device.queue.submit([cmd.finish()]);

            // testBuf.mapAsync(GPUBufferUsage.MAP_READ).then(() => {
            //     const data = new Uint32Array(testBuf.getMappedRange().slice(0));
            //     console.log(data);
            //     testBuf.destroy();
            // });
        }

        const { index, offset, buffer, model } = pipeline.allocUniform();

        this.uniformIndex = index;
        this.modelGroup = device.createBindGroup({
            layout: GCloth.renderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer,
                        offset,
                        size: Float32Array.BYTES_PER_ELEMENT * 16 * 2,
                    },
                },
            ],
        });

        this.transform = new Transform(model);
        this.transform.update();
        this.transform.updateInverse();
    }

    draw(pass: GPURenderPassEncoder) {
        pass.setBindGroup(0, this.modelGroup);
        pass.setVertexBuffer(0, this.pointBuffer);
        pass.setIndexBuffer(this.ibo, 'uint32');
        pass.drawIndexed(this.totalIndices);
    }

    free() {
        this.pipeline.freeUniformIndex(this.uniformIndex);
        this.ibo?.destroy();

        this.pointBuffer.destroy();
        this.constUB.destroy();
        this.dimUB.destroy();
        this.vectorUB.destroy();
    }

    simulate(pass: GPUComputePassEncoder) {
        // Shrinked grid
        const GridX = Math.ceil(this.dimension[0] / 14);
        const GridY = Math.ceil(this.dimension[1] / 14);

        pass.setBindGroup(0, this.s_computeGroup);
        pass.setBindGroup(1, this.s_uniformGroup);

        // TODO: oversample

        pass.dispatch(GridX, GridY);
    }

    update(pass: GPUComputePassEncoder) {
        pass.setBindGroup(0, this.u_computeGroup);
        pass.setBindGroup(1, this.u_uniformGroup);
        // Workgroup size is 256
        pass.dispatch(Math.ceil((this.dimension[0] * this.dimension[1]) / 256));
    }
}
