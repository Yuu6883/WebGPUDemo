import Renderer, { checkDevice, GDevice } from './base';

import GBufferVertWGSL from '../shaders/gbuffer.vert.wgsl';
import GBufferFragWGSL from '../shaders/gbuffer.frag.wgsl';
import QuadVertWGSL from '../shaders/quad.vert.wgsl';
import DeferredFragWGSL from '../shaders/deferred.pbr.frag.wgsl';
import PointLight from './pointlight';
import { Renderable, RenderPass } from './interfaces';
import Cloth from '../cloth/cloth';
import Particles, { GParticle } from '../particles/particles';
import StaticMesh from './staticmesh';

export const GBuffer: {
    ready: boolean;
    views: GPUTextureView[];
    basePipeline: GPURenderPipeline;
    basePassVertShader: GPUShaderModule;
    basePassFragShader: GPUShaderModule;
    deferredPipeline: GPURenderPipeline;
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
    deferredPassDesc: null,
    texGroup: null,
    dimGroup: null,
    lightGroup: null,
    viewGroup: null,
};

export class DeferredPass implements RenderPass {
    private posnorm: GPUTexture;
    private albedo: GPUTexture;

    private configUB: GPUBuffer;
    private lightNum = new Uint32Array([0]);
    private lightBuf = new Float32Array(DeferredPass.MAX_LIGHTS * PointLight.STRIDE);

    static readonly MAX_LIGHTS = 1024;
    static readonly MAX_MESHES = 4096;

    private clothDTUB: GPUBuffer;
    private particleDTUB: GPUBuffer;
    private camUB: GPUBuffer;
    private dimUB: GPUBuffer;
    private lightSB: GPUBuffer;
    private modelUB: GPUBuffer;

    public readonly lightDrawList: PointLight[] = [];
    public readonly meshDrawList: Renderable[] = [];
    public readonly clothDrawList: Cloth[] = [];
    public readonly particlesDrawList: Particles[] = [];

    private readonly freeModelUniformIndices: number[] = new Array(
        DeferredPass.MAX_MESHES,
    );

    private modelBufs: Float32Array;

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

    async init() {
        checkDevice();
        if (GBuffer.ready) return;

        const screen = GDevice.screen;
        const screenSize2D = [screen.width, screen.height];
        const device = GDevice.device;

        for (let i = 0; i < DeferredPass.MAX_MESHES; i++)
            this.freeModelUniformIndices[i] = i;

        const perModelFloats =
            device.limits.minUniformBufferOffsetAlignment /
            Float32Array.BYTES_PER_ELEMENT;

        this.modelBufs = new Float32Array(perModelFloats * DeferredPass.MAX_MESHES);

        // Create textures
        this.posnorm = device.createTexture({
            size: [...screenSize2D, 2],
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            format: 'rgba16float',
        });

        this.albedo = device.createTexture({
            size: screenSize2D,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            format: 'bgra8unorm',
        });

        GBuffer.views = [
            this.posnorm.createView({
                baseArrayLayer: 0,
                arrayLayerCount: 1,
                dimension: '2d',
            }),
            this.posnorm.createView({
                baseArrayLayer: 1,
                arrayLayerCount: 1,
                dimension: '2d',
            }),
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
                DeferredPass.MAX_LIGHTS,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // 2 * mat4x4<f32> * MAX_MODELS?
        this.modelUB = device.createBuffer({
            size: device.limits.minUniformBufferOffsetAlignment * DeferredPass.MAX_MESHES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // f32
        this.clothDTUB = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.particleDTUB = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        //  mat4x4<f32> + vec3<f32>
        this.camUB = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT * (16 + 4),
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

        GBuffer.basePipeline = await device.createRenderPipelineAsync({
            layout: 'auto',
            vertex: {
                module: GBuffer.basePassVertShader,
                entryPoint: 'main',
                buffers: [DeferredPass.VertexLayout],
            },
            fragment: {
                module: GBuffer.basePassFragShader,
                entryPoint: 'main',
                targets: [
                    // position
                    { format: 'rgba16float' },
                    // normal
                    { format: 'rgba16float' },
                    // albedo
                    { format: 'bgra8unorm' },
                ],
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

        GBuffer.deferredPipeline = await device.createRenderPipelineAsync({
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

        GBuffer.deferredPassDesc = {
            colorAttachments: [
                {
                    // view is acquired and set in render loop.
                    view: undefined,
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
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

        GBuffer.ready = true;
    }

    resize(width: number, height: number) {
        const device = GDevice.device;
        if (!GBuffer.ready) return;

        if (this.posnorm.width !== width || this.posnorm.height !== height) {
            this.posnorm.destroy();
            this.posnorm = device.createTexture({
                size: [width, height, 2],
                format: 'rgba16float',
                usage:
                    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            });
        }

        if (this.albedo.width !== width || this.albedo.height !== height) {
            this.albedo.destroy();
            this.albedo = device.createTexture({
                size: [width, height],
                format: 'bgra8unorm',
                usage:
                    GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            });
        }

        GBuffer.views = [
            this.posnorm.createView({
                baseArrayLayer: 0,
                arrayLayerCount: 1,
                dimension: '2d',
            }),
            this.posnorm.createView({
                baseArrayLayer: 1,
                arrayLayerCount: 1,
                dimension: '2d',
            }),
            this.albedo.createView(),
        ];

        const texLayout = device.createBindGroupLayout({
            entries: GBuffer.views.map((_, binding) => ({
                binding,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: 'unfilterable-float' },
            })),
        });

        GBuffer.texGroup = device.createBindGroup({
            layout: texLayout,
            entries: GBuffer.views.map((resource, binding) => ({
                binding,
                resource,
            })),
        });
    }

    addMesh(mesh: StaticMesh) {
        this.meshDrawList.push(mesh);
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
            layout: GBuffer.basePipeline.getBindGroupLayout(0),
        };
    }

    freeUniformIndex(index: number) {
        this.freeModelUniformIndices.push(index);
    }

    async render(dt: number, now: number, output: GPUTextureView, camBuf: Float32Array) {
        if (!GBuffer.ready) return;

        const device = GDevice.device;
        const queue = device.queue;

        queue.writeBuffer(this.camUB, 0, camBuf.buffer);
        queue.writeBuffer(this.configUB, 0, this.lightNum);
        queue.writeBuffer(this.lightSB, 0, this.lightBuf);
        queue.writeBuffer(this.modelUB, 0, this.modelBufs.buffer);

        const cmd = device.createCommandEncoder();
        const cmds: GPUCommandEncoder[] = [cmd];

        const SAMPLE_STEP = 1;

        queue.writeBuffer(this.clothDTUB, 0, new Float32Array([SAMPLE_STEP * 0.001]));
        queue.writeBuffer(this.particleDTUB, 0, new Float32Array([1 / 60]));

        if (GParticle.ready && this.particlesDrawList.length) {
            const StagePass = cmd.beginComputePass();
            StagePass.setPipeline(GParticle.stagePipeline);
            for (const particles of this.particlesDrawList)
                particles.stage(dt, StagePass);
            StagePass.end();

            const UpdatePass = cmd.beginComputePass();
            UpdatePass.setPipeline(GParticle.updatePipeline);
            UpdatePass.setBindGroup(3, GParticle.dtGroup);
            for (const particles of this.particlesDrawList) particles.update(UpdatePass);
            UpdatePass.end();
        }

        const basePassDesc: GPURenderPassDescriptor = {
            colorAttachments: GBuffer.views.map((view, i) => ({
                view,
                clearValue: i
                    ? { r: 0, g: 0, b: 0, a: 0 }
                    : {
                          r: Number.MAX_VALUE,
                          g: Number.MAX_VALUE,
                          b: Number.MAX_VALUE,
                          a: Number.MAX_VALUE,
                      },
                loadOp: 'clear',
                storeOp: 'store',
            })),
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

        // GBuffer base pass
        const GBufferPass = cmd.beginRenderPass(basePassDesc);

        GBufferPass.setViewport(0, 0, GDevice.screen.width, GDevice.screen.height, 0, 1);

        // Draw meshes
        GBufferPass.setPipeline(GBuffer.basePipeline);
        GBufferPass.setBindGroup(1, GBuffer.viewGroup);

        for (const mesh of this.meshDrawList) mesh.draw(GBufferPass);

        if (GParticle.ready) {
            GBufferPass.setPipeline(GParticle.renderPipeline);
            GBufferPass.setBindGroup(1, GParticle.viewGroup);
            for (const particples of this.particlesDrawList) particples.draw(GBufferPass);
        }

        GBufferPass.end();

        // Render to swapchain texture
        GBuffer.deferredPassDesc.colorAttachments[0].view = output;

        // Deferred lighting pass
        const DeferredPass = cmd.beginRenderPass(GBuffer.deferredPassDesc);

        DeferredPass.setPipeline(GBuffer.deferredPipeline);
        DeferredPass.setBindGroup(0, GBuffer.texGroup);
        DeferredPass.setBindGroup(1, GBuffer.lightGroup);
        DeferredPass.setBindGroup(2, GBuffer.dimGroup);
        DeferredPass.draw(6);
        DeferredPass.end();

        queue.submit(cmds.map(c => c.finish()));

        GBuffer.deferredPassDesc.colorAttachments[0].view = null;

        const t1 = this.particlesDrawList.map(f => f.postUpdate());
        const t2 = this.clothDrawList.map(c => c.postUpdate());
        await Promise.all(t1.concat(t2));
    }
}
