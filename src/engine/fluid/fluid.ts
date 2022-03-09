import { ReadonlyVec3, vec3 } from 'gl-matrix';
import Transform from '../core/transform';
import { CubePositions } from '../primitives/cube';
import { checkDevice, GDevice } from '../render/base';
import { Renderable, RenderPass } from '../render/interfaces';

import FluidWGSL from '../shaders/fluid.wgsl';

export interface FluidConstants {
    max_num: number;
}

export const GFluid: {
    ready: boolean;
    renderPipeline: GPURenderPipeline;
    simPipeline: GPUComputePipeline;
    updatePipeline: GPUComputePipeline;
    dtGroup: GPUBindGroup;
    viewGroup: GPUBindGroup;
    debugViewGroup: GPUBindGroup;
    cubeBuf: GPUBuffer;
} = {
    ready: false,
    renderPipeline: null,
    simPipeline: null,
    updatePipeline: null,
    dtGroup: null,
    viewGroup: null,
    debugViewGroup: null,
    cubeBuf: null,
};

export default class Fluid implements Renderable {
    private readonly uniformIndex: number;
    private readonly modelGroup: GPUBindGroup;
    private readonly sphereGroup: GPUBindGroup;
    private readonly debuModelGroup: GPUBindGroup;
    private readonly pass: RenderPass;

    public readonly transform: Transform;

    // s_ for simulation
    private readonly s_computeGroup: GPUBindGroup;
    private readonly s_uniformGroup: GPUBindGroup;

    // u_ for update
    private readonly u_computeGroup: GPUBindGroup;
    private readonly u_uniformGroup: GPUBindGroup;

    private readonly particleBuf: GPUBuffer;
    private readonly sphereUB: GPUBuffer;
    private readonly boxUB: GPUBuffer;
    private readonly vectorUB: GPUBuffer;

    // vec3<f32> x4 NOTE: vec3 is padded to 16 bytes
    static readonly STRIDE = Float32Array.BYTES_PER_ELEMENT * 4;

    private count = 1;
    private readonly max_num: number;

    static async initPipeline(camUB: GPUBuffer, dtUB: GPUBuffer) {
        checkDevice();
        const device = GDevice.device;

        GFluid.cubeBuf = device.createBuffer({
            size: CubePositions.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });

        new Float32Array(GFluid.cubeBuf.getMappedRange()).set(CubePositions);
        GFluid.cubeBuf.unmap();

        const shader = device.createShaderModule({
            code: FluidWGSL,
        });

        GFluid.renderPipeline = await device.createRenderPipeline({
            vertex: {
                module: shader,
                entryPoint: 'vert_main',
                buffers: [],
            },
            fragment: {
                module: shader,
                entryPoint: 'frag_main',
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

        GFluid.viewGroup = device.createBindGroup({
            layout: GFluid.renderPipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: camUB,
                    },
                },
            ],
        });

        // GFluid.dtGroup = device.createBindGroup({
        //     layout: GFluid.updatePipeline.getBindGroupLayout(2),
        //     entries: [
        //         {
        //             binding: 0,
        //             resource: {
        //                 buffer: dtUB,
        //             },
        //         },
        //     ],
        // });

        GFluid.ready = true;
    }

    constructor(pass: RenderPass, constants: FluidConstants) {
        checkDevice();
        this.pass = pass;
        const device = GDevice.device;

        this.max_num = constants.max_num;

        this.sphereUB = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT * 2,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        const R = 0.5;
        // TODO: calculate R based on density
        new Float32Array(this.sphereUB.getMappedRange())[0] = R;
        this.sphereUB.unmap();

        this.boxUB = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT * 6,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.particleBuf = device.createBuffer({
            size: constants.max_num * Fluid.STRIDE,
            usage:
                GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });

        const { index, offset, buffer, model } = pass.allocUniform();

        this.uniformIndex = index;
        this.modelGroup = device.createBindGroup({
            layout: GFluid.renderPipeline.getBindGroupLayout(0),
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

        this.sphereGroup = device.createBindGroup({
            layout: GFluid.renderPipeline.getBindGroupLayout(2),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.sphereUB,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.particleBuf,
                    },
                },
            ],
        });

        this.transform = new Transform(model);
        this.transform.scale = [1, 1, 1];
        this.transform.update();
        this.transform.updateInverse();
    }

    draw(pass: GPURenderPassEncoder) {
        pass.setBindGroup(0, this.modelGroup);
        pass.setBindGroup(2, this.sphereGroup);
        pass.draw(36, this.count); // 36 vertices per cube (sphere)
    }

    free() {
        this.pass.freeUniformIndex(this.uniformIndex);

        this.particleBuf.destroy();
        this.sphereUB.destroy();
        this.boxUB.destroy();
        // this.vectorUB.destroy();
    }
}
