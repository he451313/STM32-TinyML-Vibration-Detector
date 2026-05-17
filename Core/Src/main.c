/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "i2c.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "network.h"       // X-CUBE-AI 核心標頭檔
#include "network_data.h"  // 模型權重資料結構
#include "ssd1306.h"       // 新增 OLED 驅動
#include "ssd1306_fonts.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */


// --- MPU6500 暫存器與地址設定 ---
static const uint8_t mpu6500_addr = 0x68 << 1;

// --- 模型相關變數 ---
ai_handle network = AI_HANDLE_NULL;
AI_ALIGNED(32) ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

// AI 輸入與輸出快取區 (50點 * 3軸 = 150 個 float)
AI_ALIGNED(32) ai_float ai_in_data[AI_NETWORK_IN_1_SIZE];
AI_ALIGNED(32) ai_float ai_out_data[AI_NETWORK_OUT_1_SIZE];

// 定義 AI 的輸入與輸出結構體
ai_buffer ai_input;
ai_buffer ai_output;

// --- 滑動視窗緩衝區 ---
#define WINDOW_SIZE 50
#define STEP_SIZE 10
float window_buffer[WINDOW_SIZE][3];
int sample_count = 0;

// --- MinMaxScaler 參數 (請根據你 Python 的實測數據微調) ---
float x_min = -19236.0f, x_max = -10264.0f;
float y_min = 184.0f, y_max = 3788.0f;
float z_min = -4460.0f, z_max = -2244.0f;

// --- 判斷異常的門檻值 ---
const float ANOMALY_THRESHOLD = 0.17f;


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
int _write(int file, char *ptr, int len) {
    // 使用 HAL 函式庫將字串透過 huart2 傳送出去，設定超時時間為 HAL_MAX_DELAY
    HAL_UART_Transmit(&huart2, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    return len;
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_I2C1_Init();
  /* USER CODE BEGIN 2 */


  // 1. MPU6500 喚醒
  uint8_t pwr_mgmt_1_reg = 0x6B;
  uint8_t wake_up_cmd = 0x00;
  HAL_I2C_Mem_Write(&hi2c1, mpu6500_addr, pwr_mgmt_1_reg, 1, &wake_up_cmd, 1, 100);
  printf("MPU6500 已喚醒...\r\n");

  // 2. 初始化 X-CUBE-AI 網路
  ai_error ai_err;
  
  // 建立 AI 執行個體
  ai_err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
  if (ai_err.type != AI_ERROR_NONE) {
      printf("AI 網路建立失敗！\r\n");
      while(1);
  }

  // 設定啟動參數與記憶體配置
  const ai_network_params params = {
      AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
      AI_NETWORK_DATA_ACTIVATIONS(activations)
  };

  if (!ai_network_init(network, &params)) {
      printf("AI 網路初始化失敗！\r\n");
      while(1);
  }

  // 設定輸入/輸出緩衝區格式
  ai_input = ai_network_inputs_get(network, NULL)[0];
  ai_output = ai_network_outputs_get(network, NULL)[0];
  
  ai_input.data = AI_HANDLE_PTR(ai_in_data);
  ai_output.data = AI_HANDLE_PTR(ai_out_data);

  printf("X-CUBE-AI 初始化成功。開始監控馬達震動...\r\n");
  HAL_Delay(1000);

  // 3. 初始化 OLED 螢幕
  ssd1306_Init();
  ssd1306_Fill(Black);
  ssd1306_SetCursor(10, 20);
  ssd1306_WriteString("AI System", Font_11x18, White);
  ssd1306_SetCursor(25, 40);
  ssd1306_WriteString("Ready!", Font_7x10, White);
  ssd1306_UpdateScreen();
  HAL_Delay(1000); // 開機畫面停留一秒


  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    uint8_t accel_data[6];
    int16_t raw_x, raw_y, raw_z;

    // 1. 從 MPU6500 讀取實時數據
    if (HAL_I2C_Mem_Read(&hi2c1, mpu6500_addr, 0x3B, 1, accel_data, 6, 100) != HAL_OK) {
        printf("I2C 通訊中斷。嘗試重啟 I2C...\r\n");
        
        // I2C 關掉再重開
        HAL_I2C_DeInit(&hi2c1);
        HAL_Delay(10);
        HAL_I2C_Init(&hi2c1);
        continue; // 跳過這次，重新從頭讀取
    }
    raw_x = (int16_t)(accel_data[0] << 8 | accel_data[1]);
    raw_y = (int16_t)(accel_data[2] << 8 | accel_data[3]);
    raw_z = (int16_t)(accel_data[4] << 8 | accel_data[5]);

    // 2. 將數據放入滑動視窗緩衝區，同時在底層完成 MinMaxScaler (0~1)
    window_buffer[sample_count][0] = ((float)raw_x - x_min) / (x_max - x_min);
    window_buffer[sample_count][1] = ((float)raw_y - y_min) / (y_max - y_min);
    window_buffer[sample_count][2] = ((float)raw_z - z_min) / (z_max - z_min);
    sample_count++;

    // 3. 當緩衝區集滿 50 筆數據，觸發 AI 推論
    if (sample_count >= WINDOW_SIZE) {
        
        // 將 2D 視窗資料壓平成 1D 陣列傳給 AI 輸入區
        int idx = 0;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            ai_in_data[idx++] = window_buffer[i][0];
            ai_in_data[idx++] = window_buffer[i][1];
            ai_in_data[idx++] = window_buffer[i][2];
        }
        // 防呆：重新強制作業視窗與 AI 緩衝區對齊 ===
        ai_input.data = AI_HANDLE_PTR(ai_in_data);
        ai_output.data = AI_HANDLE_PTR(ai_out_data);

        // 4. 執行 AI 推論 (讓 Autoencoder 嘗試重建訊號)
        ai_i32 n_batch = ai_network_run(network, &ai_input, &ai_output);
        if (n_batch != 1) {
            printf("AI 結論出錯！\r\n");
        }

        // 5. 在 C 語言中計算 Reconstruction Loss (MSE)
        float total_squared_error = 0.0f;
        for (int i = 0; i < AI_NETWORK_IN_1_SIZE; i++) {
            float error = ai_in_data[i] - ai_out_data[i];
            total_squared_error += error * error;
        }
        float mse_score = total_squared_error / AI_NETWORK_IN_1_SIZE;

        // 6. 依據在Threshold (0.170) 進行異常告警
        // 把小數點放大 10000 倍變成整數，避開 STM32 不支援 %f 
        int mse_int = (int)(mse_score * 10000.0f);
        int threshold_int = (int)(ANOMALY_THRESHOLD * 10000.0f); 

        // 建立一個字串緩衝區來顯示分數
        char score_str[20];
        sprintf(score_str, "MSE: %d", mse_int);

        // 防呆：如果 Z 軸讀數完全是 0，代表 I2C 可能斷線
        if (raw_z == 0 && raw_x == 0) {
            printf("讀不到感測器數據\r\n");
            ssd1306_Fill(Black);
            ssd1306_SetCursor(10, 20);
            ssd1306_WriteString("SENSOR ERR", Font_11x18, White);
            ssd1306_UpdateScreen();
        } 
        else if (mse_int > threshold_int) {
            // 印出放大 10000 倍的整數分數，並附上當下 Z 軸原始數據方便觀察
            printf("震動異常 MSE分數: %d / 10000 | 原始Z軸: %d \r\n", mse_int, raw_z);

            // 螢幕顯示：ABNORMAL
            ssd1306_Fill(Black);
            ssd1306_SetCursor(15, 10);
            ssd1306_WriteString("ABNORMAL!", Font_11x18, White);
            ssd1306_SetCursor(25, 40);
            ssd1306_WriteString(score_str, Font_7x10, White);
            ssd1306_UpdateScreen();
        } 
        else {
            printf("[✅ 正常] 運轉平穩. MSE分數: %d / 10000 | 原始Z軸: %d \r\n", mse_int, raw_z);

            // 螢幕顯示：NORMAL
            ssd1306_Fill(Black);
            ssd1306_SetCursor(25, 10);
            ssd1306_WriteString("NORMAL", Font_11x18, White);
            ssd1306_SetCursor(25, 40);
            ssd1306_WriteString(score_str, Font_7x10, White);
            ssd1306_UpdateScreen();
        }
        // ==================================================
        // 7. 實現滑動視窗步長 (STEP_SIZE = 10)：將後面 40 筆資料往前移，騰出 10 筆新空間
        memmove(&window_buffer[0], &window_buffer[STEP_SIZE], (WINDOW_SIZE - STEP_SIZE) * sizeof(window_buffer[0]));
        
        // 重置採樣計數器！這行功能為避免引發 HardFault
        sample_count = WINDOW_SIZE - STEP_SIZE;

    }

    // 控制採樣頻率，與 Python 採集時保持一致 (50ms 採樣一次)
    HAL_Delay(50);
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
