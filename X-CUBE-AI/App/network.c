/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-17T20:45:38+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x5077d370ed9640338321e230c991ff58"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-05-17T20:45:38+0800"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_keras_tensor0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 150, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  gemm_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  nl_4_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  gemm_5_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  nl_5_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  nl_6_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  nl_8_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 150, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  gemm_4_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9600, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  gemm_4_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  gemm_5_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  gemm_5_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  gemm_6_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9600, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  gemm_7_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 150, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  gemm_4_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_4_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  gemm_4_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_4_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  gemm_4_weights, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 150, 64, 1, 1), AI_STRIDE_INIT(4, 4, 600, 38400, 38400),
  1, &gemm_4_weights_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  gemm_5_bias, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gemm_5_bias_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  gemm_5_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gemm_5_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  gemm_5_weights, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 64, 16, 1, 1), AI_STRIDE_INIT(4, 4, 256, 4096, 4096),
  1, &gemm_5_weights_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_bias, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_6_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_6_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  gemm_6_weights, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 16, 64, 1, 1), AI_STRIDE_INIT(4, 4, 64, 4096, 4096),
  1, &gemm_6_weights_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_bias, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 150, 1, 1), AI_STRIDE_INIT(4, 4, 4, 600, 600),
  1, &gemm_7_bias_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 150, 1, 1), AI_STRIDE_INIT(4, 4, 4, 600, 600),
  1, &gemm_7_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  gemm_7_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 64, 150, 1, 1), AI_STRIDE_INIT(4, 4, 256, 38400, 38400),
  1, &gemm_7_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  nl_4_nl_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &nl_4_nl_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  nl_5_nl_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &nl_5_nl_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  nl_6_nl_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &nl_6_nl_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  nl_8_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 150, 1, 1), AI_STRIDE_INIT(4, 4, 4, 600, 600),
  1, &nl_8_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_keras_tensor0_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 50), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &serving_default_keras_tensor0_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_keras_tensor0_output0, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 150, 1, 1), AI_STRIDE_INIT(4, 4, 4, 600, 600),
  1, &serving_default_keras_tensor0_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_8_layer, 8,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &nl_8_chain,
  NULL, &nl_8_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_6_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_7_weights, &gemm_7_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_7_layer, 7,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_7_chain,
  NULL, &nl_8_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_6_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_6_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_6_nl_layer, 6,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_6_nl_chain,
  NULL, &gemm_7_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_5_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_6_weights, &gemm_6_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_6_layer, 6,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_6_chain,
  NULL, &nl_6_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_5_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_5_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_5_nl_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_5_nl_chain,
  NULL, &gemm_6_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_5_weights, &gemm_5_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_5_layer, 5,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_5_chain,
  NULL, &nl_5_nl_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_4_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_4_nl_layer, 4,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &nl_4_nl_chain,
  NULL, &gemm_5_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_keras_tensor0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_4_weights, &gemm_4_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_4_layer, 4,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_4_chain,
  NULL, &nl_4_nl_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 86168, 1, 1),
    86168, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 856, 1, 1),
    856, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_keras_tensor0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_8_output),
  &gemm_4_layer, 0x8b49a01e, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 86168, 1, 1),
      86168, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 856, 1, 1),
      856, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_keras_tensor0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_8_output),
  &gemm_4_layer, 0x8b49a01e, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_keras_tensor0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    serving_default_keras_tensor0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_4_output_array.data = AI_PTR(g_network_activations_map[0] + 600);
    gemm_4_output_array.data_start = AI_PTR(g_network_activations_map[0] + 600);
    nl_4_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_4_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_5_output_array.data = AI_PTR(g_network_activations_map[0] + 256);
    gemm_5_output_array.data_start = AI_PTR(g_network_activations_map[0] + 256);
    nl_5_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_5_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_6_output_array.data = AI_PTR(g_network_activations_map[0] + 64);
    gemm_6_output_array.data_start = AI_PTR(g_network_activations_map[0] + 64);
    nl_6_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 600);
    nl_6_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 600);
    gemm_7_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_7_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_8_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_8_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    gemm_4_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_4_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    gemm_4_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    gemm_4_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_4_bias_array.data = AI_PTR(g_network_weights_map[0] + 38400);
    gemm_4_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 38400);
    gemm_5_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_5_weights_array.data = AI_PTR(g_network_weights_map[0] + 38656);
    gemm_5_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 38656);
    gemm_5_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_5_bias_array.data = AI_PTR(g_network_weights_map[0] + 42752);
    gemm_5_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 42752);
    gemm_6_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_6_weights_array.data = AI_PTR(g_network_weights_map[0] + 42816);
    gemm_6_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 42816);
    gemm_6_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_6_bias_array.data = AI_PTR(g_network_weights_map[0] + 46912);
    gemm_6_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 46912);
    gemm_7_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_7_weights_array.data = AI_PTR(g_network_weights_map[0] + 47168);
    gemm_7_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 47168);
    gemm_7_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_7_bias_array.data = AI_PTR(g_network_weights_map[0] + 85568);
    gemm_7_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 85568);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 23186,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x8b49a01e,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 23186,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x8b49a01e,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

