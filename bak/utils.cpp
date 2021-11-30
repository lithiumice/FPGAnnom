#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// ASCII lib from (https://www.jianshu.com/p/1f58a0ebf5d9)
const char codeLib[] =
    "@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.   ";

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