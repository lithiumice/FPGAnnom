#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include "CL/opencl.h"
// #include "AOCLUtils/aocl_utils.h"

#include "nnom.h"
#include "image.h"
#include "weights.h"

nnom_model_t *model;
// ASCII lib from (https://www.jianshu.com/p/1f58a0ebf5d9)
const char codeLib[] = "@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.   ";

void print_img(int8_t *buf)
{
	for (int y = 0; y < 28; y++)
	{
		for (int x = 0; x < 28; x++)
		{
			int index = 69 / 127.0 * (127 - buf[y * 28 + x]);
			if (index > 69)
				index = 69;
			if (index < 0)
				index = 0;
			printf("%c", codeLib[index]);
			printf("%c", codeLib[index]);
		}
		printf("\n");
	}
}

void nn_stat()
{
	model_stat(model);
	printf("Total Memory cost (Network and NNoM): %d\n", nnom_mem_stat());
}

int main(int argc, char **argv)
{
	uint32_t tick, time;
	uint32_t predic_label;
	float prob;
	int32_t index = atoi(argv[1]);

	model = nnom_model_create();
	model_run(model);

	if (index >= TOTAL_IMAGE || argc != 2)
	{
		printf("Please input image number within %d\n", TOTAL_IMAGE - 1);
		return 0;
	}

	printf("\nprediction start.. \n");
	// tick = rt_tick_get();

	// copy data and do prediction
	memcpy(nnom_input_data, (int8_t *)&img[index][0], 784);
	nnom_predict(model, &predic_label, &prob);
	// time = rt_tick_get() - tick;

	//print original image to console
	print_img((int8_t *)&img[index][0]);

	printf("Time: %d tick\n", time);
	printf("Truth label: %d\n", label[index]);
	printf("Predicted label: %d\n", predic_label);
	printf("Probability: %d%%\n", (int)(prob * 100));

	nn_stat();
	return 0;
}

void cleanup()
{
	
}
