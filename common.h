#include <string>
typedef void(*comp_func)(double *, double *, double*, double*, double*, int*, int, int, int, double*, double*, double*);
void add_function(comp_func f, std::string name);

