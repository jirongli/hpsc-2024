#include <cstdlib>
#include <cstdio>
#include <fstream>

using namespace std;

const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;
const int M = 1024;
const double dx = 2. / (nx - 1);
const double dy = 2. / (ny - 1);
const double dt = .01;
const double rho = 1.;
const double nu = .02;

dim3 block(M, 1);
dim3 grid((nx+M-1)/M, ny);

__global__ void compute_b(double *u, double *v, double *b){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = j * nx + i;
    if (i>=1 && i<nx-1 && j>=1 && j<ny-1){
        b[index] = rho * (1 / dt *
                    ((u[index+1] - u[index-1]) / (2 * dx) + (v[index+nx] - v[index-nx]) / (2 * dy)) - 
                    ((u[index+1] - u[index-1]) / (2 * dx)) * ((u[index+1] - u[index-1]) / (2 * dx)) - 
                    2 * ((u[index+nx] - u[index-nx]) / (2 * dy) * (v[index+1] - v[index-1]) / (2 * dx)) -
                    ((v[index+nx] - v[index-nx]) / (2 * dy)) * ((v[index+nx] - v[index-nx]) / (2 * dy)));
    }
}

__global__ void update_pn(double *p, double *pn){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = j * nx + i;
    if (i<nx && j<ny)
        pn[index] = p[index];
}

__global__ void compute_p(double *p, double *pn, double *b){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = j * nx + i;
    if (i>=1 && i<nx-1 && j>=1 && j<ny-1){
        p[index] = (dy * dy * (pn[index+1] + pn[index-1]) + 
                   dx * dx * (pn[index+nx] + pn[index-nx]) - 
                   b[index] * dx * dx * dy * dy ) 
                   / (2 * ( dx * dx + dy * dy ));
    }
}

__global__ void boundary_condition_p(double *p){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = j * nx + i;
    if (i==0 && j<ny) p[index] = p[index+1];
    if (i==nx-1 && j<ny) p[index] = p[index-1];
    if (j==0 && i<nx) p[index] = p[index+nx];
    if (j==ny-1 && i<nx) p[index] = 0;
}

__global__ void update_un_vn(double *u, double *v, double *un, double *vn){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = j * nx + i;
    if (i<nx && j<ny){
        un[index] = u[index];
        vn[index] = v[index];
    }
}

__global__ void compute_u_v(double *u, double *v, double *un, double *vn, double *p){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = j * nx + i;
    if (i>=1 && i<nx-1 && j>=1 && j<ny){
        u[index] = un[index] - un[index] * dt / dx * (un[index] - un[index-1])
                                - un[index] * dt / dy * (un[index] - un[index-nx])
                                - dt / (2. * rho * dx) * (p[index+1] - p[index-1])
                                + nu * dt / (dx*dx) * (un[index+1] -2*un[index] + un[index-1])
                                + nu * dt / (dy*dy) * (un[index+nx] -2*un[index] + un[index-nx]);
        v[index] = vn[index] - vn[index] * dt / dx * (vn[index] - vn[index-1])
                                - vn[index] * dt / dy * (vn[index] - vn[index-nx])
                                - dt / (2. * rho * dy) * (p[index+nx] - p[index-nx])           
                                + nu * dt / (dx*dx) * (vn[index+1] -2*vn[index] + vn[index-1])
                                + nu * dt / (dy*dy) * (vn[index+nx] -2*vn[index] + vn[index-nx]);
    }
}

__global__ void boundary_condition_u_v(double *u, double *v){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = j * nx + i;
    if (i==0 && j<ny){
        u[index] = 0;
        v[index] = 0;
    }
    if (i==nx-1 && j<ny){
        u[index] = 0;
        v[index] = 0;
    }
    if (j==0 && i<nx){
        u[index] = 0;
        v[index] = 0;
    }
    if (j==ny-1 && i<nx){
        u[index] = 1;
        v[index] = 0;
    }
}

int main (){
    double *u, *v, *p, *b, *un, *vn, *pn;
    cudaMallocManaged(&u, nx*ny*sizeof(double));
    cudaMallocManaged(&v, nx*ny*sizeof(double));
    cudaMallocManaged(&p, nx*ny*sizeof(double));
    cudaMallocManaged(&b, nx*ny*sizeof(double));
    cudaMallocManaged(&un, nx*ny*sizeof(double));
    cudaMallocManaged(&vn, nx*ny*sizeof(double));
    cudaMallocManaged(&pn, nx*ny*sizeof(double));

    cudaMemset(u, 0, nx*ny*sizeof(double));
    cudaMemset(v, 0, nx*ny*sizeof(double));
    cudaMemset(p, 0, nx*ny*sizeof(double));
    cudaMemset(b, 0, nx*ny*sizeof(double));
    cudaMemset(un, 0, nx*ny*sizeof(double));
    cudaMemset(vn, 0, nx*ny*sizeof(double));
    cudaMemset(pn, 0, nx*ny*sizeof(double));
    
    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    for(int n=0; n<nt; n++){
        // Compute b[j][i]
        compute_b<<<grid, block>>>(u, v, b);
        cudaDeviceSynchronize();
        for(int it=0; it<nit; it++){
            update_pn<<<grid, block>>>(p, pn);
            cudaDeviceSynchronize();
            // Compute p[j][i]
            compute_p<<<grid, block>>>(p, pn, b);
            cudaDeviceSynchronize();
            // Compute p[j][0], p[j][nx-1], p[0][i], p[ny-1][i]
            boundary_condition_p<<<grid, block>>>(p);
            cudaDeviceSynchronize();
        }
        update_un_vn<<<grid, block>>>(u, v, un, vn);
        cudaDeviceSynchronize();
        // Compute u[j][i], v[j][i]
        compute_u_v<<<grid, block>>>(u, v, un, vn, p);
        cudaDeviceSynchronize();
        // Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
        // Compute u[0][i], u[ny-1][i], v[0][i], v[ny-1][i]
        boundary_condition_u_v<<<grid, block>>>(u, v);
        cudaDeviceSynchronize();
        if (n % 10 == 0){
            for (int j=0; j<ny; j++)
                for (int i=0; i<nx; i++)
                    ufile << u[j*nx+i] << " ";
                ufile << "\n";
            for (int j=0; j<ny; j++)
                for (int i=0; i<nx; i++)
                    vfile << v[j*nx+i] << " ";
                vfile << "\n";
            for (int j=0; j<ny; j++)
                for (int i=0; i<nx; i++)
                pfile << p[j*nx+i] << " ";
            pfile << "\n";
        }
    }
    ufile.close();
    vfile.close();
    pfile.close();

    cudaFree(u);
    cudaFree(v);
    cudaFree(b);
    cudaFree(p);
    cudaFree(un);
    cudaFree(vn);
    cudaFree(pn);
    
    return 0;
}