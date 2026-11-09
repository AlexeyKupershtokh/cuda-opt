#ifndef RDSCLL__
#define RDSCLL__

#define rdtscll(val) { \
unsigned int __a,__d; \
asm volatile("rdtsc" : "=a" (__a), "=d" (__d)); \
(val) = ((unsigned long)__a) | (((unsigned long)__d)<<32); \
} 


#endif

