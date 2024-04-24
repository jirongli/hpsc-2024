#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], j[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j[i] = i;
  }
  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  __m512 jvec = _mm512_load_ps(j);
  for(int i=0; i<N; i++) {
    __m512 ivec = _mm512_set1_ps(i);
    __m512 xivec = _mm512_set1_ps(x[i]);
    __m512 yivec = _mm512_set1_ps(y[i]);
    //mask
    __mmask16 mask = _mm512_cmp_ps_mask(ivec,jvec,_CMP_NEQ_OQ);
    __m512 zerovec = _mm512_setzero_ps();
    //rx, ry
    __m512 rxvec = _mm512_sub_ps(xivec,xvec);
    __m512 ryvec = _mm512_sub_ps(yivec,yvec);
    //r
    __m512 rvec = _mm512_sqrt_ps(_mm512_add_ps(_mm512_mul_ps(rxvec,rxvec),_mm512_mul_ps(ryvec,ryvec)));
    //fx, fy
    __m512 rsqvec = _mm512_rsqrt14_ps(rvec);
    __m512 r1vec = _mm512_mul_ps(rsqvec,rsqvec);
    __m512 r3vec = _mm512_mul_ps(r1vec,_mm512_mul_ps(r1vec,r1vec));
    __m512 fxivec = _mm512_mask_blend_ps(mask,zerovec,_mm512_mul_ps(_mm512_mul_ps(rxvec,mvec),r3vec));
    __m512 fyivec = _mm512_mask_blend_ps(mask,zerovec,_mm512_mul_ps(_mm512_mul_ps(ryvec,mvec),r3vec));
    fx[i] -= _mm512_reduce_add_ps(fxivec);
    fy[i] -= _mm512_reduce_add_ps(fyivec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
