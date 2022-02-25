import { checkDevice, GDevice } from './base';

import GBufferVertWGSL from '../shaders/gbuffer.vert.wgsl';
import GBufferFragWGSL from '../shaders/gbuffer.frag.wgsl';
import QuadVertWGSL from '../shaders/quad.vert.wgsl';
import DeferredFragWGSL from '../shaders/deferred.frag.wgsl';
import PointLight from './pointlight';
import { Renderable } from './pass';
import Cloth, { GCloth } from '../cloth/cloth';

export const GBuffer: {
    ready: boolean;
    views: GPUTextureView[];
    basePipeline: GPURenderPipeline;
    basePassVertShader: GPUShaderModule;
    basePassFragShader: GPUShaderModule;
    deferredPipeline: GPURenderPipeline;
    basePassDesc: GPURenderPassDescriptor;
    deferredPassDesc: GPURenderPassDescriptor;
    texGroup: GPUBindGroup;
    dimGroup: GPUBindGroup;
    lightGroup: GPUBindGroup;
    viewGroup: GPUBindGroup;
} = {
    ready: false,
    views: null,
    basePipeline: null,
    basePassVertShader: null,
    basePassFragShader: null,
    deferredPipeline: null,
    basePassDesc: null,
    deferredPassDesc: null,
    texGroup: null,
    dimGroup: null,
    lightGroup: null,
    viewGroup: null,
};

export class DeferredPipeline {
    private readonly posnorm: GPUTexture;
    private readonly albedo: GPUTexture;
    private readonly depth: GPUTexture;

    private readonly configUB: GPUBuffer;
    private readonly lightNum = new Uint32Array([0]);
    private readonly lightBuf = new Float32Array(
        DeferredPipeline.MAX_LIGHTS * PointLight.STRIDE,
    );

    static readonly MAX_LIGHTS = 1024;
    static readonly MAX_MESHES = 1024;

    private readonly dtUB: GPUBuffer;
    private readonly camUB: GPUBuffer;
    private readonly dimUB: GPUBuffer;
    private readonly lightSB: GPUBuffer;
    private readonly modelUB: GPUBuffer;

    public readonly lightDrawList: PointLight[] = [];
    public readonly meshDrawList: Renderable[] = [];
    public readonly clothDrawList: Cloth[] = [];

    private readonly freeModelUniformIndices: number[] = new Array(
        DeferredPipeline.MAX_MESHES,
    );

    private readonly modelBufs: Float32Array;

    public static readonly VertexLayout: GPUVertexBufferLayout = {
        arrayStride: Float32Array.BYTES_PER_ELEMENT * 8,
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
                offset: Float32Array.BYTES_PER_ELEMENT * 3,
                format: 'float32x3',
            },
            {
                // uv
                shaderLocation: 2,
                offset: Float32Array.BYTES_PER_ELEMENT * 6,
                format: 'float32x2',
            },
        ],
    };

    constructor() {
        checkDevice();
        if (GBuffer.ready) return;
        GBuffer.ready = true;

        const screen = GDevice.screen;
        const screenSize2D = [screen.width, screen.height];
        const device = GDevice.device;

        // Create textures
        this.posnorm = device.createTexture({
            size: [...screenSize2D, 2],
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            format: 'rgba32float',
        });

        this.albedo = device.createTexture({
            size: screenSize2D,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            format: 'bgra8unorm',
        });

        this.depth = device.createTexture({
            size: screenSize2D,
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        GBuffer.views = [
            this.posnorm.createView({ baseArrayLayer: 0, arrayLayerCount: 1 }),
            this.posnorm.createView({ baseArrayLayer: 1, arrayLayerCount: 1 }),
            this.albedo.createView(),
        ];

        // Create buffers
        this.configUB = device.createBuffer({
            size: Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        new Uint32Array(this.configUB.getMappedRange())[0] = this.lightNum[0];
        this.configUB.unmap();

        this.lightSB = device.createBuffer({
            size:
                Float32Array.BYTES_PER_ELEMENT *
                PointLight.STRIDE *
                DeferredPipeline.MAX_LIGHTS,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // 2 * mat4x4<f32> * MAX_MODELS?
        this.modelUB = device.createBuffer({
            size:
                device.limits.minUniformBufferOffsetAlignment *
                DeferredPipeline.MAX_MESHES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // f32
        this.dtUB = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        //  mat4x4<f32>
        this.camUB = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT * 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // vec2<f32>
        this.dimUB = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT * 2,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        GBuffer.basePassVertShader = device.createShaderModule({
            code: GBufferVertWGSL,
        });

        GBuffer.basePassFragShader = device.createShaderModule({
            code: GBufferFragWGSL,
        });

        GBuffer.basePipeline = device.createRenderPipeline({
            vertex: {
                module: GBuffer.basePassVertShader,
                entryPoint: 'main',
                buffers: [DeferredPipeline.VertexLayout],
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

        const texLayout = device.createBindGroupLayout({
            entries: GBuffer.views.map((_, binding) => ({
                binding,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: 'unfilterable-float' },
            })),
        });

        GBuffer.texGroup = device.createBindGroup({
            layout: texLayout,
            entries: GBuffer.views.map((resource, binding) => ({ binding, resource })),
        });

        const lightLayout = device.createBindGroupLayout({
            entries: [
                // Light data
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'storage',
                    },
                },
                // Light num
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                    },
                },
            ],
        });

        GBuffer.lightGroup = device.createBindGroup({
            layout: lightLayout,
            entries: [
                { binding: 0, resource: { buffer: this.lightSB } },
                { binding: 1, resource: { buffer: this.configUB } },
            ],
        });

        const sizeLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                    },
                },
            ],
        });

        const quadVertShader = device.createShaderModule({
            code: QuadVertWGSL,
        });

        const deferredLightsFragShader = device.createShaderModule({
            code: DeferredFragWGSL,
        });

        GBuffer.deferredPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [texLayout, lightLayout, sizeLayout],
            }),
            vertex: {
                module: quadVertShader,
                entryPoint: 'main',
            },
            fragment: {
                module: deferredLightsFragShader,
                entryPoint: 'main',
                targets: [
                    {
                        format: GDevice.format,
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',
            },
        });

        GBuffer.basePassDesc = {
            colorAttachments: GBuffer.views.map((view, i) => ({
                view,
                loadValue: i
                    ? { r: 0, g: 0, b: 0, a: 0 }
                    : {
                          r: Number.MAX_VALUE,
                          g: Number.MAX_VALUE,
                          b: Number.MAX_VALUE,
                          a: Number.MAX_VALUE,
                      },
                loadOp: 'load',
                storeOp: 'store',
            })),
            depthStencilAttachment: {
                view: this.depth.createView(),
                depthLoadValue: 1.0,
                depthStoreOp: 'store',
                stencilLoadValue: 0,
                stencilStoreOp: 'store',
            },
        };

        GBuffer.deferredPassDesc = {
            colorAttachments: [
                {
                    // view is acquired and set in render loop.
                    view: undefined,
                    loadValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'load',
                    storeOp: 'store',
                },
            ],
        };

        GBuffer.viewGroup = device.createBindGroup({
            layout: GBuffer.basePipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.camUB,
                    },
                },
            ],
        });

        GBuffer.dimGroup = device.createBindGroup({
            layout: sizeLayout,
            entries: [{ binding: 0, resource: { buffer: this.dimUB } }],
        });

        device.queue.writeBuffer(this.dimUB, 0, new Float32Array(screenSize2D));

        for (let i = 0; i < DeferredPipeline.MAX_MESHES; i++)
            this.freeModelUniformIndices[i] = i;

        const perModelFloats =
            device.limits.minUniformBufferOffsetAlignment /
            Float32Array.BYTES_PER_ELEMENT;

        this.modelBufs = new Float32Array(perModelFloats * DeferredPipeline.MAX_MESHES);

        Cloth.initPipeline(this.camUB, this.dtUB);
    }

    updateSize() {
        // TODO
    }

    updateLight() {
        for (let i = 0; i < this.lightDrawList.length; i++) {
            const light = this.lightDrawList[i];
            this.lightBuf.set(light.buffer, PointLight.STRIDE * i);
        }
        this.lightNum[0] = this.lightDrawList.length;
    }

    allocUniform() {
        const index = this.freeModelUniformIndices.shift();
        const align = GDevice.device.limits.minUniformBufferOffsetAlignment;
        const perModelFloats = align / Float32Array.BYTES_PER_ELEMENT;

        const begin = index * perModelFloats;
        const end = index * perModelFloats + 16 * 2;

        return {
            index,
            offset: align * index,
            buffer: this.modelUB,
            model: this.modelBufs.subarray(begin, end),
        };
    }

    freeUniformIndex(index: number) {
        this.freeModelUniformIndices.push(index);
    }

    async render(
        dt: number,
        now: number,
        output: GPUTextureView,
        viewProjection: Float32Array,
    ) {
        const device = GDevice.device;
        const queue = device.queue;

        queue.writeBuffer(this.camUB, 0, viewProjection);
        queue.writeBuffer(this.configUB, 0, this.lightNum);
        queue.writeBuffer(this.lightSB, 0, this.lightBuf);
        queue.writeBuffer(this.modelUB, 0, this.modelBufs);

        const cmd = device.createCommandEncoder();

        const SAMPLE_STEP = 1;

        // Milliseconds
        queue.writeBuffer(this.dtUB, 0, new Float32Array([SAMPLE_STEP * 0.001]));

        let loop = 0;
        while (Cloth.sampleTime < now) {
            Cloth.sampleTime += SAMPLE_STEP * Cloth.sampleRate;
            loop++;
        }

        for (let i = 0; i < Math.min(50, loop); i++) {
            // Cloth compute pass
            const ClothForcePass = cmd.beginComputePass();
            ClothForcePass.setPipeline(GCloth.forceCalcPipeline);
            for (const cloth of this.clothDrawList) cloth.simulate(ClothForcePass);
            ClothForcePass.endPass();

            // Cloth update pass
            const ClothUpdatePass = cmd.beginComputePass();
            ClothUpdatePass.setBindGroup(2, GCloth.dtGroup);
            ClothUpdatePass.setPipeline(GCloth.updatePipeline);
            for (const cloth of this.clothDrawList) cloth.update(ClothUpdatePass);
            ClothUpdatePass.endPass();
        }

        const ClothNormalPass = cmd.beginComputePass();
        ClothNormalPass.setPipeline(GCloth.normalCalcPipeline);
        for (const cloth of this.clothDrawList) cloth.recalcNormals(ClothNormalPass);
        ClothNormalPass.endPass();

        // GBuffer base pass
        const GBufferPass = cmd.beginRenderPass(GBuffer.basePassDesc);

        GBufferPass.setViewport(0, 0, GDevice.screen.width, GDevice.screen.height, 0, 1);

        // Draw meshes
        GBufferPass.setPipeline(GBuffer.basePipeline);
        GBufferPass.setBindGroup(1, GBuffer.viewGroup);

        for (const mesh of this.meshDrawList) mesh.draw(GBufferPass);

        // Draw clothes / debug normal
        if (Cloth.debug) {
            GBufferPass.setPipeline(GCloth.normalDebugPipeline);
            GBufferPass.setBindGroup(1, GCloth.debugViewGroup);
            for (const cloth of this.clothDrawList) cloth.debug(GBufferPass);
        } else {
            GBufferPass.setPipeline(GCloth.renderPipeline);
            GBufferPass.setBindGroup(1, GCloth.viewGroup);
            for (const cloth of this.clothDrawList) cloth.draw(GBufferPass);
        }

        GBufferPass.endPass();

        // Render to swapchain texture
        GBuffer.deferredPassDesc.colorAttachments[0].view = output;

        // Deferred lighting pass
        const DeferredPass = cmd.beginRenderPass(GBuffer.deferredPassDesc);

        DeferredPass.setPipeline(GBuffer.deferredPipeline);
        DeferredPass.setBindGroup(0, GBuffer.texGroup);
        DeferredPass.setBindGroup(1, GBuffer.lightGroup);
        DeferredPass.setBindGroup(2, GBuffer.dimGroup);
        DeferredPass.draw(6);
        DeferredPass.endPass();

        queue.submit([cmd.finish()]);

        for (const cloth of this.clothDrawList) await cloth.postUpdate();

        GBuffer.deferredPassDesc.colorAttachments[0].view = null;
    }
}
