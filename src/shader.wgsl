enable dual_source_blending;

struct VertexInput {
    @builtin(vertex_index) vertex_idx: u32,
    @location(0) pos: vec2<i32>,
    @location(1) dim: u32,
    @location(2) uv: u32,
    @location(3) color: u32,
    @location(4) content_type_with_srgb: u32,
    @location(5) depth: f32,
}

struct VertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) content_type: u32,
};

struct Params {
    screen_resolution: vec2<u32>,
    _pad: vec2<u32>,
};

@group(0) @binding(0)
var color_atlas_texture: texture_2d<f32>;

@group(0) @binding(1)
var mask_atlas_texture: texture_2d<f32>;

@group(0) @binding(2)
var atlas_sampler: sampler;

@group(1) @binding(0)
var<uniform> params: Params;

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    } else {
        return pow((c + 0.055) / 1.055, 2.4);
    }
}

@vertex
fn vs_main(in_vert: VertexInput) -> VertexOutput {
    var pos = in_vert.pos;
    let width = in_vert.dim & 0xffffu;
    let height = (in_vert.dim & 0xffff0000u) >> 16u;
    let color = in_vert.color;
    var uv = vec2<u32>(in_vert.uv & 0xffffu, (in_vert.uv & 0xffff0000u) >> 16u);
    let v = in_vert.vertex_idx;

    let corner_position = vec2<u32>(
        in_vert.vertex_idx & 1u,
        (in_vert.vertex_idx >> 1u) & 1u,
    );

    let corner_offset = vec2<u32>(width, height) * corner_position;

    uv = uv + corner_offset;
    pos = pos + vec2<i32>(corner_offset);

    var vert_output: VertexOutput;

    vert_output.position = vec4<f32>(
        2.0 * vec2<f32>(pos) / vec2<f32>(params.screen_resolution) - 1.0,
        in_vert.depth,
        1.0,
    );

    vert_output.position.y *= -1.0;

    let content_type = in_vert.content_type_with_srgb & 0xffffu;
    let srgb = (in_vert.content_type_with_srgb & 0xffff0000u) >> 16u;

    switch srgb {
        case 0u: {
            vert_output.color = vec4<f32>(
                f32((color & 0x00ff0000u) >> 16u) / 255.0,
                f32((color & 0x0000ff00u) >> 8u) / 255.0,
                f32(color & 0x000000ffu) / 255.0,
                f32((color & 0xff000000u) >> 24u) / 255.0,
            );
        }
        case 1u: {
            vert_output.color = vec4<f32>(
                srgb_to_linear(f32((color & 0x00ff0000u) >> 16u) / 255.0),
                srgb_to_linear(f32((color & 0x0000ff00u) >> 8u) / 255.0),
                srgb_to_linear(f32(color & 0x000000ffu) / 255.0),
                f32((color & 0xff000000u) >> 24u) / 255.0,
            );
        }
        default: {}
    }

    var dim: vec2<u32> = vec2(0u);
    switch content_type {
        case 0u: {
            dim = textureDimensions(color_atlas_texture);
            break;
        }
        case 1u: {
            dim = textureDimensions(mask_atlas_texture);
            break;
        }
        case 2u: {
            dim = textureDimensions(color_atlas_texture);
            break;
        }
        default: {}
    }

    vert_output.content_type = content_type;

    vert_output.uv = vec2<f32>(uv) / vec2<f32>(dim);

    return vert_output;
}

struct FragmentOutput {
    @location(0) @blend_src(0) color: vec4<f32>,
    @location(0) @blend_src(1) mask: vec4<f32>,
}

// Gamma correction helpers from swash_demo
fn luma(color: vec3<f32>) -> f32 {
    return color.x * 0.25 + color.y * 0.72 + color.z * 0.075;
}

fn gamma_correct(luma_val: f32, alpha: f32, gamma: f32, contrast: f32) -> f32 {
    let inverse_luma = 1.0 - luma_val;
    let inverse_alpha = 1.0 - alpha;
    let g = pow(luma_val * alpha + inverse_luma * inverse_alpha, gamma);
    let a_raw = (g - inverse_luma) / (luma_val - inverse_luma + 0.0001); // Add epsilon to avoid division by zero
    let a = a_raw + ((1.0 - a_raw) * contrast * a_raw);
    return clamp(a, 0.0, 1.0);
}

fn gamma_correct_subpx(text_color: vec3<f32>, mask: vec3<f32>) -> vec3<f32> {
    let l = luma(text_color);
    let inverse_luma = 1.0 - l;
    // Adjust gamma and contrast to match Chrome's bolder appearance
    // Higher contrast = bolder text
    let gamma = mix(1.0 / 1.15, 1.0 / 2.2, inverse_luma);
    let contrast = mix(0.3, 1.0, inverse_luma);
    return vec3<f32>(
        gamma_correct(l, mask.x, gamma, contrast),
        gamma_correct(l, mask.y, gamma, contrast),
        gamma_correct(l, mask.z, gamma, contrast)
    );
}

// Apply FreeType-style 5-tap LCD filter to subpixel coverage
// Filter weights: {16, 64, 112, 64, 16} / 208 = {0.077, 0.308, 0.538, 0.308, 0.077}
// IMPORTANT: Atlas stores inverted coverage (white=no text, black=text), so we must invert before filtering
fn apply_lcd_filter(uv: vec2<f32>) -> vec3<f32> {
    let atlas_size = vec2<f32>(textureDimensions(color_atlas_texture));
    let texel_size = 1.0 / atlas_size;

    // Sample 5 horizontal pixels and invert them (1.0 - x to get actual coverage)
    let c0 = 1.0 - textureSampleLevel(color_atlas_texture, atlas_sampler, uv + vec2<f32>(-2.0, 0.0) * texel_size, 0.0).rgb;
    let c1 = 1.0 - textureSampleLevel(color_atlas_texture, atlas_sampler, uv + vec2<f32>(-1.0, 0.0) * texel_size, 0.0).rgb;
    let c2 = 1.0 - textureSampleLevel(color_atlas_texture, atlas_sampler, uv, 0.0).rgb;
    let c3 = 1.0 - textureSampleLevel(color_atlas_texture, atlas_sampler, uv + vec2<f32>(1.0, 0.0) * texel_size, 0.0).rgb;
    let c4 = 1.0 - textureSampleLevel(color_atlas_texture, atlas_sampler, uv + vec2<f32>(2.0, 0.0) * texel_size, 0.0).rgb;

    // Apply the 5-tap filter weights to actual coverage values
    let filtered = (c0 * 16.0 + c1 * 64.0 + c2 * 112.0 + c3 * 64.0 + c4 * 16.0) / 208.0;

    // Return filtered coverage (still non-inverted, ready for gamma correction)
    return filtered;
}

@fragment
fn fs_main(in_frag: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;

    switch in_frag.content_type {
        case 0u: {
            // Color glyphs
            let color = textureSampleLevel(color_atlas_texture, atlas_sampler, in_frag.uv, 0.0);
            output.color = color;
            output.mask = vec4<f32>(1.0);
        }
        case 1u: {
            // Grayscale mask
            let mask_val = textureSampleLevel(mask_atlas_texture, atlas_sampler, in_frag.uv, 0.0).x;
            output.color = in_frag.color;
            output.mask = vec4<f32>(mask_val);
        }
        case 2u: {
            // Subpixel rendering with LCD filter and gamma correction
            // Coverage atlas stores inverted values (white=no text, black=text)
            // First apply the 5-tap LCD filter to smooth the coverage
            let coverage = apply_lcd_filter(in_frag.uv);
            // Then apply gamma correction like swash_demo
            let corrected_mask = gamma_correct_subpx(in_frag.color.rgb, coverage);
            output.color = vec4<f32>(in_frag.color.rgb, 1.0);
            output.mask = vec4<f32>(corrected_mask, 1.0);
        }
        default: {
            output.color = vec4<f32>(0.0);
            output.mask = vec4<f32>(0.0);
        }
    }

    return output;
}
