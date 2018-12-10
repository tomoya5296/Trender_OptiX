/* 
 * Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <optixu/optixu_math_namespace.h>                                        
using namespace optix;

static __device__ __inline__ float cosTheta(const float3& w) {
	return w.z;
}

static __device__ __inline__ float cos2Theta(const float3& w) {
	double c = cosTheta(w);
	return c * c;
}

static __device__ __inline__ float sinTheta(const float3& w) {
	return sqrt(max(0.0, 1.0 - cos2Theta(w)));
}

static __device__ __inline__ float tanTheta(const float3& w) {
	return sinTheta(w) / cosTheta(w);
}

static __device__ __inline__ float cosPhi(const float3& w) {
	if (w.z == 1.0) return 0.0;

	return w.x / sinTheta(w);
}

static __device__ __inline__ float cos2Phi(const float3& w) {
	double c = cosPhi(w);
	return c * c;
}

static __device__ __inline__ float sinPhi(const float3& w) {
	if (w.z == 1.0) return 0.0;

	return w.y / sinTheta(w);
}

static __device__ __inline__ float sin2Phi(const float3& w) {
	double s = sinPhi(w);
	return s * s;
}

static __device__ __inline__ float3 exp(const float3& x)
{
	return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

struct ParallelogramLight                                                        
{                                                                                
    optix::float3 corner;                                                          
    optix::float3 v1, v2;                                                          
    optix::float3 normal;                                                          
    optix::float3 emission;                                                        
};                                                                               

