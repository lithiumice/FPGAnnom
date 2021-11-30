#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nnom.h"
#include "image.h"
#include "weights.h"
#include "utils.h"
#include "cl_utils.h"

// void nn_stat()
// {
//   model_stat(model);
//   printf("Total Memory cost (Network and NNoM): %d\n", nnom_mem_stat());
// }

// extern "C"
// {
int main(int argc, char **argv)
{
  nnom_model_t *model;
  printf("hello world\n");

  uint32_t tick, time;
  uint32_t predic_label;
  float prob;
  int32_t index = atoi(argv[1]);
  char *argv0 = argv[0];
  char *argv1 = argv[1];
  char *argv2 = argv[2];
  char *argv3 = argv[3];

  if (index < 0 || index >= TOTAL_IMAGE)
  {
    printf("Please input image number within %d\n", TOTAL_IMAGE - 1);
    printf("usage: ./host N[N: image index]\n");
    return 0;
  }

  init_opencl();
  // init_var();
  // init_problem();
  // run_problem();

  // // tick = rt_tick_get();

  // // copy data and do prediction

  // time = rt_tick_get() - tick;

  //print original image to console
  print_img((int8_t *)&img[index][0]);

  model = nnom_model_create();
  // model_run(model);

  printf("\nprediction start.. \n");
  memcpy(nnom_input_data, (int8_t *)&img[index][0], 784);
  nnom_predict(model, &predic_label, &prob);

  printf("Time: %d tick\n", time);
  printf("Truth label: %d\n", label[index]);
  printf("Predicted label: %d\n", predic_label);
  printf("Probability: %d%%\n", (int)(prob * 100));

  // nn_stat();
  cleanup();

  return 0;
}
// }