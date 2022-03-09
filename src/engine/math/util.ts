export const clamp = (n: number, min: number, max: number) =>
    n < min ? min : n > max ? max : n;

export const normalize_nd = (vec: Float32Array) => {
    let v = 0;
    for (let i = 0; i < vec.length; i++) {
        v += vec[i] * vec[i];
    }
    v = Math.sqrt(v);
    return vec.map(n => n / v);
};
