#include <stdio.h>

extern isize zig_quantize_model(char *model_path, unsigned char bits);

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_model>\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    zig_quantize_model((char *)path, 8);
    printf("Quantization completed\n");
    return 0;
}
