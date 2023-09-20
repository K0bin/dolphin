// Copyright 2016 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Common/Hash.h"

#include "VideoBackends/Vulkan/ShaderCompiler.h"

#include <cstddef>
#include <string>

#include "VideoBackends/Vulkan/VulkanContext.h"
#include "VideoCommon/Spirv.h"

namespace Vulkan::ShaderCompiler
{

    static const char REPLACEMENT_SHADER[] = R"(

                                                                                                      // Target GLSL 4.5.
                                                                                                      #version 450 core
                                                                                                      #define ATTRIBUTE_LOCATION(x) layout(location = x)
                                                                                                      #define FRAGMENT_OUTPUT_LOCATION(x) layout(location = x)
                                                                                                      #define FRAGMENT_OUTPUT_LOCATION_INDEXED(x, y) layout(location = x, index = y)
                                                                                                      #define UBO_BINDING(packing, x) layout(packing, set = 0, binding = (x - 1))
                                                                                                      #define SAMPLER_BINDING(x) layout(set = 1, binding = x)
                                                                                                      #define TEXEL_BUFFER_BINDING(x) layout(set = 1, binding = (x + 8))
                                                                                                      #define SSBO_BINDING(x) layout(std430, set = 2, binding = x)
                                                                                                      #define INPUT_ATTACHMENT_BINDING(x, y, z) layout(set = x, binding = y, input_attachment_index = z)
                                                                                                      #define VARYING_LOCATION(x) layout(location = x)
                                                                                                      #define FORCE_EARLY_Z layout(early_fragment_tests) in

                                                                                                      // Metal framebuffer fetch helpers.
                                                                                                      #define FB_FETCH_VALUE subpassLoad(in_ocol0)

                                                                                                      // hlsl to glsl function translation
                                                                                                      #define API_VULKAN 1
                                                                                                      #define float2 vec2
                                                                                                      #define float3 vec3
                                                                                                      #define float4 vec4
                                                                                                      #define uint2 uvec2
                                                                                                      #define uint3 uvec3
                                                                                                      #define uint4 uvec4
                                                                                                      #define int2 ivec2
                                                                                                      #define int3 ivec3
                                                                                                      #define int4 ivec4
                                                                                                      #define frac fract
                                                                                                      #define lerp mix

                                                                                                      // These were changed in Vulkan
                                                                                                      #define gl_VertexID gl_VertexIndex
                                                                                                      #define gl_InstanceID gl_InstanceIndex
                                                                                                    // Pixel Shader for TEV stages
                                                                                                    // 1 TEV stages, 1 texgens, 0 IND stages
                                                                                                    int idot(int3 x, int3 y)
                                                                                                    {
                                                                                                    	int3 tmp = x * y;
                                                                                                    	return tmp.x + tmp.y + tmp.z;
                                                                                                    }
                                                                                                    int idot(int4 x, int4 y)
                                                                                                    {
                                                                                                    	int4 tmp = x * y;
                                                                                                    	return tmp.x + tmp.y + tmp.z + tmp.w;
                                                                                                    }

                                                                                                    int  iround(float  x) { return int (round(x)); }
                                                                                                    int2 iround(float2 x) { return int2(round(x)); }
                                                                                                    int3 iround(float3 x) { return int3(round(x)); }
                                                                                                    int4 iround(float4 x) { return int4(round(x)); }

                                                                                                    SAMPLER_BINDING(0) uniform sampler2DArray samp[8];

                                                                                                    UBO_BINDING(std140, 1) uniform PSBlock {
                                                                                                    	int4 color[4];
                                                                                                    	int4 k[4];
                                                                                                    	int4 alphaRef;
                                                                                                    	int4 texdim[8];
                                                                                                    	int4 czbias[2];
                                                                                                    	int4 cindscale[2];
                                                                                                    	int4 cindmtx[6];
                                                                                                    	int4 cfogcolor;
                                                                                                    	int4 cfogi;
                                                                                                    	float4 cfogf;
                                                                                                    	float4 cfogrange[3];
                                                                                                    	float4 czslope;
                                                                                                    	float2 cefbscale;
                                                                                                    	uint  bpmem_genmode;
                                                                                                    	uint  bpmem_alphaTest;
                                                                                                    	uint  bpmem_fogParam3;
                                                                                                    	uint  bpmem_fogRangeBase;
                                                                                                    	uint  bpmem_dstalpha;
                                                                                                    	uint  bpmem_ztex_op;
                                                                                                    	bool  bpmem_late_ztest;
                                                                                                    	bool  bpmem_rgba6_format;
                                                                                                    	bool  bpmem_dither;
                                                                                                    	bool  bpmem_bounding_box;
                                                                                                    	uint4 bpmem_pack1[16];
                                                                                                    	uint4 bpmem_pack2[8];
                                                                                                    	int4  konstLookup[32];
                                                                                                    	bool  blend_enable;
                                                                                                    	uint  blend_src_factor;
                                                                                                    	uint  blend_src_factor_alpha;
                                                                                                    	uint  blend_dst_factor;
                                                                                                    	uint  blend_dst_factor_alpha;
                                                                                                    	bool  blend_subtract;
                                                                                                    	bool  blend_subtract_alpha;
                                                                                                    	bool  logic_op_enable;
                                                                                                    	uint  logic_op_mode;
                                                                                                    	uint  time_ms;
                                                                                                    };

                                                                                                    #define bpmem_combiners(i) (bpmem_pack1[(i)].xy)
                                                                                                    #define bpmem_tevind(i) (bpmem_pack1[(i)].z)
                                                                                                    #define bpmem_iref(i) (bpmem_pack1[(i)].w)
                                                                                                    #define bpmem_tevorder(i) (bpmem_pack2[(i)].x)
                                                                                                    #define bpmem_tevksel(i) (bpmem_pack2[(i)].y)
                                                                                                    #define samp_texmode0(i) (bpmem_pack2[(i)].z)
                                                                                                    #define samp_texmode1(i) (bpmem_pack2[(i)].w)


                                                                                                    int4 sampleTexture(uint texmap, in sampler2DArray tex, int2 uv, int layer) {
                                                                                                      float size_s = float(texdim[texmap].x * 128);
                                                                                                      float size_t = float(texdim[texmap].y * 128);
                                                                                                      float3 coords = float3(float(uv.x) / size_s, float(uv.y) / size_t, layer);
                                                                                                      return iround(255.0 * texture(tex, coords));
                                                                                                    }
                                                                                                    #define CUSTOM_SHADER_API_VERSION 1;
                                                                                                    const uint CUSTOM_SHADER_LIGHTING_ATTENUATION_TYPE_NONE = 0u;
                                                                                                    const uint CUSTOM_SHADER_LIGHTING_ATTENUATION_TYPE_POINT = 1u;
                                                                                                    const uint CUSTOM_SHADER_LIGHTING_ATTENUATION_TYPE_DIR = 2u;
                                                                                                    const uint CUSTOM_SHADER_LIGHTING_ATTENUATION_TYPE_SPOT = 3u;
                                                                                                    struct CustomShaderLightData
                                                                                                    {
                                                                                                    	float3 position;
                                                                                                    	float3 direction;
                                                                                                    	float3 color;
                                                                                                    	uint attenuation_type;
                                                                                                    	float4 cosatt;
                                                                                                    	float4 distatt;
                                                                                                    };

                                                                                                    const uint CUSTOM_SHADER_TEV_STAGE_INPUT_TYPE_PREV = 0u;
                                                                                                    const uint CUSTOM_SHADER_TEV_STAGE_INPUT_TYPE_COLOR = 1u;
                                                                                                    const uint CUSTOM_SHADER_TEV_STAGE_INPUT_TYPE_TEX = 2u;
                                                                                                    const uint CUSTOM_SHADER_TEV_STAGE_INPUT_TYPE_RAS = 3u;
                                                                                                    const uint CUSTOM_SHADER_TEV_STAGE_INPUT_TYPE_KONST = 4u;
                                                                                                    const uint CUSTOM_SHADER_TEV_STAGE_INPUT_TYPE_NUMERIC = 5u;
                                                                                                    const uint CUSTOM_SHADER_TEV_STAGE_INPUT_TYPE_UNUSED = 6u;
                                                                                                    struct CustomShaderTevStageInputColor
                                                                                                    {
                                                                                                    	uint input_type;
                                                                                                    	float3 value;
                                                                                                    };

                                                                                                    struct CustomShaderTevStageInputAlpha
                                                                                                    {
                                                                                                    	uint input_type;
                                                                                                    	float value;
                                                                                                    };

                                                                                                    struct CustomShaderTevStage
                                                                                                    {
                                                                                                    	CustomShaderTevStageInputColor[4] input_color;
                                                                                                    	CustomShaderTevStageInputAlpha[4] input_alpha;
                                                                                                    	uint texmap;
                                                                                                    	float4 output_color;
                                                                                                    };

                                                                                                    struct CustomShaderData
                                                                                                    {
                                                                                                    	float3 position;
                                                                                                    	float3 normal;
                                                                                                    	float3[1] texcoord;
                                                                                                    	uint texcoord_count;
                                                                                                    	uint[8] texmap_to_texcoord_index;
                                                                                                    	CustomShaderLightData[8] lights_chan0_color;
                                                                                                    	CustomShaderLightData[8] lights_chan0_alpha;
                                                                                                    	CustomShaderLightData[8] lights_chan1_color;
                                                                                                    	CustomShaderLightData[8] lights_chan1_alpha;
                                                                                                    	float4[2] ambient_lighting;
                                                                                                    	float4[2] base_material;
                                                                                                    	uint light_chan0_color_count;
                                                                                                    	uint light_chan0_alpha_count;
                                                                                                    	uint light_chan1_color_count;
                                                                                                    	uint light_chan1_alpha_count;
                                                                                                    	CustomShaderTevStage[16] tev_stages;
                                                                                                    	uint tev_stage_count;
                                                                                                    	float4 final_color;
                                                                                                    	uint time_ms;
                                                                                                    };


                                                                                                    #define sampleTextureWrapper(texmap, uv, layer) sampleTexture(texmap, samp[texmap], uv, layer)
                                                                                                    FRAGMENT_OUTPUT_LOCATION_INDEXED(0, 0) out vec4 ocol0;
                                                                                                    FRAGMENT_OUTPUT_LOCATION_INDEXED(0, 1) out vec4 ocol1;
                                                                                                    VARYING_LOCATION(0)  in float4 colors_0;
                                                                                                    VARYING_LOCATION(1)  in float4 colors_1;
                                                                                                    VARYING_LOCATION(2)  in float3 tex0;
                                                                                                    void main()
                                                                                                    {
                                                                                                    	float4 rawpos = gl_FragCoord;
                                                                                                    	int layer = 0;
                                                                                                    	int4 c0 = color[1], c1 = color[2], c2 = color[3], prev = color[0];
                                                                                                    	int4 rastemp = int4(0, 0, 0, 0), textemp = int4(0, 0, 0, 0), konsttemp = int4(0, 0, 0, 0);
                                                                                                    	int3 comp16 = int3(1, 256, 0), comp24 = int3(1, 256, 256*256);
                                                                                                    	int alphabump=0;
                                                                                                    	int3 tevcoord=int3(0, 0, 0);
                                                                                                    	int2 wrappedcoord=int2(0,0), tempcoord=int2(0,0);
                                                                                                    	int4 tevin_a=int4(0,0,0,0),tevin_b=int4(0,0,0,0),tevin_c=int4(0,0,0,0),tevin_d=int4(0,0,0,0);

                                                                                                    	float4 col0 = colors_0;
                                                                                                    	float4 col1 = colors_1;
                                                                                                    	int2 fixpoint_uv0 = int2((tex0.z == 0.0 ? tex0.xy : tex0.xy / tex0.z) * float2(texdim[0].zw * 128));

                                                                                                    	// TEV stage 0
                                                                                                    	// indirect op
                                                                                                    	int2 indtevtrans0 = int2(0, 0);
                                                                                                    	wrappedcoord.x = fixpoint_uv0.x;
                                                                                                    	wrappedcoord.y = fixpoint_uv0.y;
                                                                                                    	tevcoord.xy = wrappedcoord + indtevtrans0;
                                                                                                    	tevcoord.xy = (tevcoord.xy << 8) >> 8;
                                                                                                    	textemp = sampleTextureWrapper(0u, tevcoord.xy, layer).rgba;
                                                                                                    	tevin_a = int4(c1.rgb, 0)&int4(255, 255, 255, 255);
                                                                                                    	tevin_b = int4(c0.rgb, c0.a)&int4(255, 255, 255, 255);
                                                                                                    	tevin_c = int4(textemp.rgb, textemp.a)&int4(255, 255, 255, 255);
                                                                                                    	tevin_d = int4(int3(0,0,0), 0);
                                                                                                    	// color combine
                                                                                                    	prev.rgb = clamp((((tevin_d.rgb)) + (((((tevin_a.rgb<<8) + (tevin_b.rgb-tevin_a.rgb)*(tevin_c.rgb+(tevin_c.rgb>>7)))) + 128)>>8)), int3(0,0,0), int3(255,255,255));
                                                                                                    	// alpha combine
                                                                                                    	prev.a = clamp((((tevin_d.a)) + (((((tevin_a.a<<8) + (tevin_b.a-tevin_a.a)*(tevin_c.a+(tevin_c.a>>7)))) + 128)>>8)), 0, 255);
                                                                                                    	prev = prev & 255;
                                                                                                        gl_SampleMask[0] = 0xFF;
                                                                                                    	if(!( (prev.a >  alphaRef.r) || (prev.a >  alphaRef.g))) {
                                                                                                    		ocol0 = float4(0.0, 0.0, 0.0, 0.0);
                                                                                                    		ocol1 = float4(0.0, 0.0, 0.0, 0.0);
                                                                                                    		//discard;
                                                                                                            gl_SampleMask[0] = 0;
                                                                                                    	}
                                                                                                    	// Hardware testing indicates that an alpha of 1 can pass an alpha test,
                                                                                                    	// but doesn't do anything in blending
                                                                                                    	if (prev.a == 1) prev.a = 0;
                                                                                                    	int zCoord = int((1.0 - rawpos.z) * 16777216.0);
                                                                                                    	zCoord = clamp(zCoord, 0, 0xFFFFFF);
                                                                                                    	ocol0.rgb = float3(prev.rgb) / 255.0;
                                                                                                    	ocol0.a = float(prev.a >> 2) / 63.0;
                                                                                                    	ocol1 = float4(0.0, 0.0, 0.0, float(prev.a) / 255.0);
                                                                                                    }

    )";

// Regarding the UBO bind points, we subtract one from the binding index because
// the OpenGL backend requires UBO #0 for non-block uniforms (at least on NV).
// This allows us to share the same shaders but use bind point #0 in the Vulkan
// backend. None of the Vulkan-specific shaders use UBOs, instead they use push
// constants, so when/if the GL backend moves to uniform blocks completely this
// subtraction can be removed.
static const char SHADER_HEADER[] = R"(
  // Target GLSL 4.5.
  #version 450 core
  #define ATTRIBUTE_LOCATION(x) layout(location = x)
  #define FRAGMENT_OUTPUT_LOCATION(x) layout(location = x)
  #define FRAGMENT_OUTPUT_LOCATION_INDEXED(x, y) layout(location = x, index = y)
  #define UBO_BINDING(packing, x) layout(packing, set = 0, binding = (x - 1))
  #define SAMPLER_BINDING(x) layout(set = 1, binding = x)
  #define TEXEL_BUFFER_BINDING(x) layout(set = 1, binding = (x + 8))
  #define SSBO_BINDING(x) layout(std430, set = 2, binding = x)
  #define INPUT_ATTACHMENT_BINDING(x, y, z) layout(set = x, binding = y, input_attachment_index = z)
  #define VARYING_LOCATION(x) layout(location = x)
  #define FORCE_EARLY_Z layout(early_fragment_tests) in

  // Metal framebuffer fetch helpers.
  #define FB_FETCH_VALUE subpassLoad(in_ocol0)

  // hlsl to glsl function translation
  #define API_VULKAN 1
  #define float2 vec2
  #define float3 vec3
  #define float4 vec4
  #define uint2 uvec2
  #define uint3 uvec3
  #define uint4 uvec4
  #define int2 ivec2
  #define int3 ivec3
  #define int4 ivec4
  #define frac fract
  #define lerp mix

  // These were changed in Vulkan
  #define gl_VertexID gl_VertexIndex
  #define gl_InstanceID gl_InstanceIndex
)";
static const char COMPUTE_SHADER_HEADER[] = R"(
  // Target GLSL 4.5.
  #version 450 core
  // All resources are packed into one descriptor set for compute.
  #define UBO_BINDING(packing, x) layout(packing, set = 0, binding = (x - 1))
  #define SAMPLER_BINDING(x) layout(set = 0, binding = (1 + x))
  #define TEXEL_BUFFER_BINDING(x) layout(set = 0, binding = (9 + x))
  #define IMAGE_BINDING(format, x) layout(format, set = 0, binding = (11 + x))

  // hlsl to glsl function translation
  #define API_VULKAN 1
  #define float2 vec2
  #define float3 vec3
  #define float4 vec4
  #define uint2 uvec2
  #define uint3 uvec3
  #define uint4 uvec4
  #define int2 ivec2
  #define int3 ivec3
  #define int4 ivec4
  #define frac fract
  #define lerp mix
)";
static const char SUBGROUP_HELPER_HEADER[] = R"(
  #extension GL_KHR_shader_subgroup_basic : enable
  #extension GL_KHR_shader_subgroup_arithmetic : enable
  #extension GL_KHR_shader_subgroup_ballot : enable
  #extension GL_KHR_shader_subgroup_shuffle : enable

  #define SUPPORTS_SUBGROUP_REDUCTION 1
  #define IS_HELPER_INVOCATION gl_HelperInvocation
  #define IS_FIRST_ACTIVE_INVOCATION (subgroupElect())
  #define SUBGROUP_MIN(value) value = subgroupMin(value)
  #define SUBGROUP_MAX(value) value = subgroupMax(value)
)";

static std::string GetShaderCode(std::string_view source, std::string_view header)
{
  std::string full_source_code;
  if (!header.empty())
  {
    constexpr size_t subgroup_helper_header_length = std::size(SUBGROUP_HELPER_HEADER) - 1;
    full_source_code.reserve(header.size() + subgroup_helper_header_length + source.size());
    full_source_code.append(header);
    if (g_vulkan_context->SupportsShaderSubgroupOperations())
      full_source_code.append(SUBGROUP_HELPER_HEADER, subgroup_helper_header_length);
    full_source_code.append(source);
  }

  u64 hash = Common::GetHash64(reinterpret_cast<const u8*>(full_source_code.c_str()), full_source_code.size(), 128);

  if (hash == 1691436099762771950 && false) {
    INFO_LOG_FMT(HOST_GPU, "REPLACING Shader: {}, hash: {}", full_source_code, hash);
    return REPLACEMENT_SHADER;
  }

  INFO_LOG_FMT(HOST_GPU, "Shader: {}, hash: {}", full_source_code, hash);

  return full_source_code;
}

static glslang::EShTargetLanguageVersion GetLanguageVersion()
{
  // Sub-group operations require Vulkan 1.1 and SPIR-V 1.3.
  if (g_vulkan_context->SupportsShaderSubgroupOperations())
    return glslang::EShTargetSpv_1_3;

  return glslang::EShTargetSpv_1_0;
}

std::optional<SPIRVCodeVector> CompileVertexShader(std::string_view source_code)
{
  return SPIRV::CompileVertexShader(GetShaderCode(source_code, SHADER_HEADER), APIType::Vulkan,
                                    GetLanguageVersion());
}

std::optional<SPIRVCodeVector> CompileGeometryShader(std::string_view source_code)
{
  return SPIRV::CompileGeometryShader(GetShaderCode(source_code, SHADER_HEADER), APIType::Vulkan,
                                      GetLanguageVersion());
}

std::optional<SPIRVCodeVector> CompileFragmentShader(std::string_view source_code)
{
  return SPIRV::CompileFragmentShader(GetShaderCode(source_code, SHADER_HEADER), APIType::Vulkan,
                                      GetLanguageVersion());
}

std::optional<SPIRVCodeVector> CompileComputeShader(std::string_view source_code)
{
  return SPIRV::CompileComputeShader(GetShaderCode(source_code, COMPUTE_SHADER_HEADER),
                                     APIType::Vulkan, GetLanguageVersion());
}
}  // namespace Vulkan::ShaderCompiler
