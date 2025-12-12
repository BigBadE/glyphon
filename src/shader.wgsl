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
    let contrast = 2.0;
    return vec3<f32>(
        gamma_correct(l, mask.x, gamma, contrast),
        gamma_correct(l, mask.y, gamma, contrast),
        gamma_correct(l, mask.z, gamma, contrast)
    );
}

// Apply 5-tap FreeType-style LCD filter for subpixel antialiasing
// Filter weights: [0x10, 0x40, 0x70, 0x40, 0x10] normalized to [0.0625, 0.25, 0.4375, 0.25, 0.0625]
fn apply_lcd_filter(uv: vec2<f32>) -> vec3<f32> {
    let dim = vec2<f32>(textureDimensions(color_atlas_texture));
    let texel_size = 1.0 / dim;

    // Sample 5 horizontal pixels
    let s0 = textureSampleLevel(color_atlas_texture, atlas_sampler, uv + vec2<f32>(-2.0 * texel_size.x, 0.0), 0.0).rgb;
    let s1 = textureSampleLevel(color_atlas_texture, atlas_sampler, uv + vec2<f32>(-1.0 * texel_size.x, 0.0), 0.0).rgb;
    let s2 = textureSampleLevel(color_atlas_texture, atlas_sampler, uv, 0.0).rgb;
    let s3 = textureSampleLevel(color_atlas_texture, atlas_sampler, uv + vec2<f32>(1.0 * texel_size.x, 0.0), 0.0).rgb;
    let s4 = textureSampleLevel(color_atlas_texture, atlas_sampler, uv + vec2<f32>(2.0 * texel_size.x, 0.0), 0.0).rgb;

    // Apply FreeType default filter weights
    return s0 * 0.0625 + s1 * 0.25 + s2 * 0.4375 + s3 * 0.25 + s4 * 0.0625;
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
            // Subpixel rendering - atlas contains LCD-filtered coverage from Swash
            //
            // WORKAROUND: Sample as .bgr instead of .rgb to correct channel swap issue.
            // Root cause: On Vulkan/Windows, when sampling from Rgba8UnormSrgb textures,
            // the R and B channels appear swapped despite correct upload format.
            // This may be a driver quirk or Vulkan/wgpu behavior on Windows.
            // Swash outputs RGB data, but when sampled it returns BGR, so we swap here.
            let coverage = textureSampleLevel(color_atlas_texture, atlas_sampler, in_frag.uv, 0.0).bgr;
            // First apply power curve to boost mid-tones
            let gamma_boosted = pow(coverage, vec3<f32>(0.6, 0.6, 0.6));
            // Then apply gamma correction with contrast
            let corrected_mask = gamma_correct_subpx(in_frag.color.rgb, gamma_boosted);
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
