import Transform from '../core/transform';
import { checkDevice, GDevice } from '../render/base';
import { Renderable, RenderPass } from '../render/interfaces';

import ParticleWGSL from '../shaders/particles.wgsl';

export interface ParticleConstants {
    max_num: number;
    max_spawn_per_frame: number;
}

export const GParticle: {
    ready: boolean;
    renderPipeline: GPURenderPipeline;
    stagePipeline: GPUComputePipeline;
    updatePipeline: GPUComputePipeline;
    viewGroup: GPUBindGroup;
    dtGroup: GPUBindGroup;
} = {
    ready: false,
    renderPipeline: null,
    stagePipeline: null,
    updatePipeline: null,
    viewGroup: null,
    dtGroup: null,
};

const rng = (min: number, max: number) => Math.random() * (max - min) + min;

export default class Particles implements Renderable {
    private readonly pass: RenderPass;

    private readonly uniformIndex: number;

    private readonly modelGroup: GPUBindGroup;
    private readonly renderGroup: GPUBindGroup;

    private readonly stageGroup: GPUBindGroup;
    private readonly updateGroup: GPUBindGroup;

    private readonly stageUniformGroup: GPUBindGroup;
    private readonly updateUniformGroup: GPUBindGroup;

    private readonly stageCameraGroup: GPUBindGroup;
    private readonly updateCameraGroup: GPUBindGroup;

    public readonly transform: Transform;

    private readonly particleBuf: GPUBuffer;
    private readonly indicesBuf: GPUBuffer;
    private readonly stageBuf: GPUBuffer;

    private readonly cpuStageBuf: Float32Array;

    private readonly paramsUB: GPUBuffer;
    private readonly sphereBuffer = new ArrayBuffer(8 * 4 + 4 * 4);

    public readonly initPos = new Float32Array([0, 25, 0]);
    public readonly initVel = new Float32Array([0, 75, 0]);
    public readonly variPos = new Float32Array([25, 25, 25]);
    public readonly variVel = new Float32Array([20, 20, 20]);
    public readonly lifeSpan = new Float32Array([15000, 1000]);
    // Air density, drag, elasticity, friction,
    public readonly coeffients = new Float32Array([1, 0.01, 0.5, 0.5]);
    public readonly wind = new Float32Array([0, 0, 0]);

    private usedListTop = 0;
    private readonly usedIndices: Uint32Array;
    private readonly freeIndices: number[] = [];
    private readonly particleLifeSpan: Float32Array;

    // vec3<f32> x4 NOTE: vec3 is padded to 16 bytes
    static readonly STRIDE = Float32Array.BYTES_PER_ELEMENT * 4 * 2;

    private count = 0;
    public radius = 1;
    public spawn_rate = 1000;

    private readonly max_num: number;
    private readonly max_spawn_per_frame: number;

    public pause = false;

    static async initPipeline(camUB: GPUBuffer, dtUB: GPUBuffer) {
        checkDevice();
        const device = GDevice.device;
        const shader = device.createShaderModule({
            code: ParticleWGSL,
        });

        GParticle.renderPipeline = await device.createRenderPipelineAsync({
            layout: 'auto',
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
                format: 'depth24plus',
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
        });

        GParticle.stagePipeline = await device.createComputePipelineAsync({
            layout: 'auto',
            compute: {
                module: shader,
                entryPoint: 'spawn',
            },
        });

        GParticle.updatePipeline = await device.createComputePipelineAsync({
            layout: 'auto',
            compute: {
                module: shader,
                entryPoint: 'update',
            },
        });

        GParticle.viewGroup = device.createBindGroup({
            layout: GParticle.renderPipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: camUB,
                    },
                },
            ],
        });

        GParticle.dtGroup = device.createBindGroup({
            layout: GParticle.updatePipeline.getBindGroupLayout(3),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: dtUB,
                    },
                },
            ],
        });

        GParticle.ready = true;
    }

    constructor(pass: RenderPass, constants: ParticleConstants) {
        checkDevice();
        this.pass = pass;
        const device = GDevice.device;

        constants.max_spawn_per_frame = Math.min(
            constants.max_num,
            constants.max_spawn_per_frame,
        );

        this.max_num = constants.max_num;
        this.max_spawn_per_frame = constants.max_spawn_per_frame;
        this.cpuStageBuf = new Float32Array(this.max_spawn_per_frame * 8);

        for (let i = 0; i < this.max_num; i++) {
            this.freeIndices.push(i);
        }

        this.paramsUB = device.createBuffer({
            size: 4 * 8 + 4 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        const view = new DataView(this.paramsUB.getMappedRange());
        view.setFloat32(0, this.radius, true);
        this.paramsUB.unmap();

        this.usedIndices = new Uint32Array(constants.max_num);
        this.particleBuf = device.createBuffer({
            size: constants.max_num * Particles.STRIDE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.indicesBuf = device.createBuffer({
            size: constants.max_num * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.stageBuf = device.createBuffer({
            size: constants.max_spawn_per_frame * Particles.STRIDE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.particleLifeSpan = new Float32Array(constants.max_num);

        const { index, offset, buffer, model } = pass.allocUniform();

        this.uniformIndex = index;
        this.modelGroup = device.createBindGroup({
            layout: GParticle.renderPipeline.getBindGroupLayout(0),
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

        this.renderGroup = device.createBindGroup({
            layout: GParticle.renderPipeline.getBindGroupLayout(2),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.paramsUB,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.particleBuf,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.indicesBuf,
                    },
                },
            ],
        });

        this.stageGroup = device.createBindGroup({
            layout: GParticle.stagePipeline.getBindGroupLayout(2),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.paramsUB,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.particleBuf,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.stageBuf,
                    },
                },
            ],
        });

        this.updateGroup = device.createBindGroup({
            layout: GParticle.updatePipeline.getBindGroupLayout(2),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.paramsUB,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.particleBuf,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.indicesBuf,
                    },
                },
            ],
        });

        // TODO: remove this garbage
        this.stageUniformGroup = device.createBindGroup({
            layout: GParticle.stagePipeline.getBindGroupLayout(0),
            entries: [],
        });

        this.updateUniformGroup = device.createBindGroup({
            layout: GParticle.updatePipeline.getBindGroupLayout(0),
            entries: [],
        });

        this.stageCameraGroup = device.createBindGroup({
            layout: GParticle.stagePipeline.getBindGroupLayout(1),
            entries: [],
        });

        this.updateCameraGroup = device.createBindGroup({
            layout: GParticle.updatePipeline.getBindGroupLayout(1),
            entries: [],
        });

        this.transform = new Transform(model);
        this.transform.scale = [1, 1, 1];
        this.transform.update();
        this.transform.updateInverse();
    }

    private filterParticles(dt: number) {
        const now = GDevice.now;

        let write_index = 0;
        for (let i = 0; i < this.count; i++) {
            if (i > write_index) this.usedIndices[write_index] = this.usedIndices[i];

            const index = this.usedIndices[i];
            if (this.particleLifeSpan[index] < now) {
                // Particle deleted
                this.particleLifeSpan[index] = 0;
                this.freeIndices.push(index);
                this.usedListTop--;
            } else {
                write_index++;
                this.particleLifeSpan[index] += dt;
            }
        }
        this.count = write_index;
    }

    private spawnParticles(dt: number) {
        const toSpawn = Math.min(~~(this.spawn_rate * dt), this.max_spawn_per_frame);
        const now = GDevice.now;

        const spawning = this.freeIndices.splice(0, toSpawn);
        this.usedIndices.set(spawning, this.usedListTop);
        this.usedListTop += spawning.length;
        this.count += spawning.length;

        for (let i = 0; i < spawning.length; i++) {
            const newIndex = spawning[i];
            // Initialize life span
            this.particleLifeSpan[newIndex] =
                now +
                rng(
                    this.lifeSpan[0] - this.lifeSpan[1],
                    this.lifeSpan[0] + this.lifeSpan[1],
                );

            // Initialize the particle to stage buffer
            const offset = i * 8;
            this.cpuStageBuf[offset + 0] = rng(
                this.initPos[0] - this.variPos[0],
                this.initPos[0] + this.variPos[0],
            );
            this.cpuStageBuf[offset + 1] = rng(
                this.initPos[1] - this.variPos[1],
                this.initPos[1] + this.variPos[1],
            );
            this.cpuStageBuf[offset + 2] = rng(
                this.initPos[2] - this.variPos[2],
                this.initPos[2] + this.variPos[2],
            );
            this.cpuStageBuf[offset + 3] = newIndex;
            this.cpuStageBuf[offset + 4] = rng(
                this.initVel[0] - this.variVel[0],
                this.initVel[0] + this.variVel[0],
            );
            this.cpuStageBuf[offset + 5] = rng(
                this.initVel[1] - this.variVel[1],
                this.initVel[1] + this.variVel[1],
            );
            this.cpuStageBuf[offset + 6] = rng(
                this.initVel[2] - this.variVel[2],
                this.initVel[2] + this.variVel[2],
            );
        }

        return spawning.length;
    }

    stage(dt: number, pass: GPUComputePassEncoder) {
        this.filterParticles(dt);

        if (this.pause) return;
        const spawned = this.spawnParticles(dt);

        const view = new DataView(this.sphereBuffer);

        view.setFloat32(0, this.radius, true);
        view.setUint32(4, this.count, true);
        view.setUint32(8, spawned, true);
        view.setFloat32(12, -9.81, true);
        view.setFloat32(16, this.coeffients[0], true);
        view.setFloat32(20, this.coeffients[1], true);
        view.setFloat32(24, this.coeffients[2], true);
        view.setFloat32(28, this.coeffients[3], true);
        view.setFloat32(32, this.wind[0], true);
        view.setFloat32(36, this.wind[1], true);
        view.setFloat32(40, this.wind[2], true);

        const queue = GDevice.device.queue;
        queue.writeBuffer(this.stageBuf, 0, this.cpuStageBuf, 0, spawned * 8);
        queue.writeBuffer(this.indicesBuf, 0, this.usedIndices, 0, this.count);
        queue.writeBuffer(this.paramsUB, 0, this.sphereBuffer);

        pass.setBindGroup(0, this.stageUniformGroup);
        pass.setBindGroup(1, this.stageCameraGroup);
        pass.setBindGroup(2, this.stageGroup);
        pass.dispatchWorkgroups(Math.ceil(spawned / 256));
    }

    update(pass: GPUComputePassEncoder) {
        if (this.pause) return;

        pass.setBindGroup(0, this.updateUniformGroup);
        pass.setBindGroup(1, this.updateCameraGroup);
        pass.setBindGroup(2, this.updateGroup);
        pass.dispatchWorkgroups(Math.ceil(this.count / 256));
    }

    async postUpdate() {}

    draw(pass: GPURenderPassEncoder) {
        pass.setBindGroup(0, this.modelGroup);
        pass.setBindGroup(2, this.renderGroup);
        pass.draw(36, this.count); // 36 vertices per cube (sphere)
    }

    free() {
        this.pass.freeUniformIndex(this.uniformIndex);

        this.particleBuf.destroy();
        this.paramsUB.destroy();
        // this.vectorUB.destroy();
    }
}
