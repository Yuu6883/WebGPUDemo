import { EntityRenderParams } from '../types';

type TextureAtlasEntry = EntityRenderParams & { offset: number };

export default class TextureAtlas {
    public scratch: HTMLCanvasElement;
    private scratchCtx: CanvasRenderingContext2D;

    public readonly entries: TextureAtlasEntry[] = [];
    public static readonly DIM = 1024;

    constructor() {
        this.scratch = document.createElement('canvas');
        this.scratchCtx = this.scratch.getContext('2d', {
            alpha: true,
        });
        this.scratch.width = TextureAtlas.DIM;
        this.scratch.height = TextureAtlas.DIM;
    }

    public async load() {
        const ctx = this.scratchCtx;
        ctx.clearRect(0, 0, this.scratch.width, this.scratch.height);

        const config: EntityRenderParams[] = await fetch('assets/entity.json').then(r =>
            r.json(),
        );
        const tasks: Promise<void>[] = [];

        const loadImage = async (url: string) => {
            const img = new Image();
            img.src = url;
            await img.decode();
            return img;
        };

        let offset = 0;
        for (const entity of config) {
            const task = (async (offset: number) => {
                for (let i = 0; i < entity.variant; i++) {
                    const o = offset + i * 3;
                    const v = (i + 1).toString().padStart(2, '0');

                    const base = await loadImage(`assets/${entity.name}-colour-${v}.png`);
                    const normal = await loadImage(
                        `assets/${entity.name}-normal-${v}.png`,
                    );
                    const rough = await loadImage(
                        `assets/${entity.name}-roughness-${v}.png`,
                    );

                    const w = base.width;
                    const h = base.height;
                    const check = (img: HTMLImageElement) => {
                        if (img.width !== w || img.height !== h) {
                            throw new Error(
                                `Image ${img.src} has wrong size: ${img.width}x${img.height} instead of ${w}x${h}`,
                            );
                        }
                        if (img.width > 64 || img.height > 64) {
                            throw new Error(
                                `Image ${img.src} is too big: ${img.width}x${img.height}`,
                            );
                        }
                    };

                    check(base);
                    check(normal);
                    check(rough);

                    const draw = (img: HTMLImageElement, o: number) => {
                        const cx = o % 16;
                        const cy = Math.floor(o / 16);
                        const ox = Math.floor((64 - w) / 2);
                        const oy = Math.floor((64 - h) / 2);
                        ctx.drawImage(img, cx * 64 + ox, cy * 64 + oy);
                    };

                    draw(base, o);
                    draw(normal, o + 1);
                    draw(rough, o + 2);
                }
            })(offset);
            tasks.push(task);
            this.entries.push({ ...entity, offset });

            offset += entity.variant * 3;
            if (offset > 16 * 16) throw new Error('Too many textures');
        }

        await Promise.all(tasks);
        // document.body.appendChild(this.scratch);
    }
}
