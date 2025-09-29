#include <stdio.h>

extern isize zig_quantize_model(char *model_path, unsigned char bits);

int main() {
    const char *path = "../../../models/dummy_model.bin";
    zig_quantize_model(path, 8);
    printf("Quantization completed\n");
    return 0;
}
