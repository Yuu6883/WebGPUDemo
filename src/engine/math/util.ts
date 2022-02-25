export const clamp = (n: number, min: number, max: number) =>
    n < min ? min : n > max ? max : n;
