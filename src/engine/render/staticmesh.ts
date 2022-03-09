import Transform from '../core/transform';
import { GDevice } from './base';
import { Renderable, RenderPass } from './interfaces';

export default class StaticMesh implements Renderable {
    public readonly transform: Transform;
    public readonly count: number;

    private readonly indexFormat: GPUIndexFormat;

    private readonly vbo: GPUBuffer;
    private readonly ibo: GPUBuffer;

    public readonly uniformIndex: number;
    private readonly modelGroup: GPUBindGroup;
    private readonly pass: RenderPass;

    constructor(
        pass: RenderPass,
        positions: Float32Array,
        normals: Float32Array,
        uvs: Float32Array,
        indices?: Uint32Array | Uint16Array,
    ) {
        if (GDevice.readyState !== 2) {
            throw new Error('GPUDevice not ready');
        }
        if (
            positions.length % 3 ||
            positions.length !== normals.length ||
            uvs.length / 2 !== normals.length / 3
        ) {
            throw new Error('StaticMesh: Buffer length mismatch');
        }

        this.pass = pass;
        const device = GDevice.device;

        const elem = positions.length / 3;
        const stride = 3 + 3 + 2;
        this.vbo = device.createBuffer({
            // position: vec3, normal: vec3, uv: vec2
            size: elem * stride * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });

        const mapped = new Float32Array(this.vbo.getMappedRange());
        for (let i = 0; i < elem; i++) {
            mapped[stride * i + 0] = positions[i * 3 + 0];
            mapped[stride * i + 1] = positions[i * 3 + 1];
            mapped[stride * i + 2] = positions[i * 3 + 2];
            mapped[stride * i + 3] = normals[i * 3 + 0];
            mapped[stride * i + 4] = normals[i * 3 + 1];
            mapped[stride * i + 5] = normals[i * 3 + 2];
            mapped[stride * i + 6] = uvs[i * 2 + 0];
            mapped[stride * i + 7] = uvs[i * 2 + 1];
        }
        this.vbo.unmap();

        if (!indices) {
            this.indexFormat = null;
            this.ibo = null;

            this.count = elem;
        } else {
            this.count = indices.length / 3;

            this.ibo = device.createBuffer({
                size: this.count,
                usage: GPUBufferUsage.INDEX,
                mappedAtCreation: true,
            });

            if (indices.BYTES_PER_ELEMENT === 2) {
                new Uint16Array(this.ibo.getMappedRange()).set(indices);
                this.indexFormat = 'uint16';
            } else if (indices.BYTES_PER_ELEMENT === 4) {
                new Uint32Array(this.ibo.getMappedRange()).set(indices);
                this.indexFormat = 'uint32';
            }
            this.ibo.unmap();
        }

        const { index, offset, buffer, model, layout } = pass.allocUniform();

        this.uniformIndex = index;
        this.modelGroup = device.createBindGroup({
            layout,
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

    free() {
        this.pass.freeUniformIndex(this.uniformIndex);
        this.vbo.destroy();
        this.ibo?.destroy();
    }

    draw(pass: GPURenderPassEncoder) {
        pass.setBindGroup(0, this.modelGroup);
        pass.setVertexBuffer(0, this.vbo);
        if (this.ibo) {
            pass.setIndexBuffer(this.ibo, this.indexFormat);
            pass.drawIndexed(this.count);
        } else {
            pass.draw(this.count);
        }
    }
}
