#include <stdio.h>

int main()
{
    auto int number = 5;
    {
        auto int number = 20;
        printf("inner number: %d", number);
    }
    printf("\n");
    printf("outer number: %d\n", number);
    return 0;
}
