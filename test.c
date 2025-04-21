#include <stdio.h>
#include <complex.h>

int main(){
  float _Complex x = 2 + 3 * I;
  double _Complex y = 2 + 3 * I;
  long double _Complex z = 2 + 3 * I;
  printf("%ld %ld %ld\n", sizeof(x), sizeof(y), sizeof(z));
}
