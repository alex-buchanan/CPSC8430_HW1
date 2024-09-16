#include <stdio.h>
#include "extern_example.h"
  
extern int var; 	
  // int var;  ->  declaration and definition
               // extern int var;  -> declaration
  
int main() 
{  
   printf("%d\n", var);
   
   return 0;
}
