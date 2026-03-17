// gibson - cyberpunk megastructure voxel renderer
// single-file C++17 / OpenGL 3.3 / GLFW
//
// TOC:
//   [1] includes + platform
//   [2] math library
//   [3] simplex noise
//   [4] catmull-rom splines
//   [5] enums + constants
//   [6] material + district tables
//   [7] WFC solver
//   [8] L-system engine
//   [9] biome layer defs
//   [10] MegaStructureGenerator
//   [11] orbital camera
//   [12] FPS camera + collision
//   [13] GLSL shaders
//   [14] renderer
//   [15] input + main loop
//   [16] seed utilities + main()

// ============================================================
// [1] INCLUDES + PLATFORM
// ============================================================
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <array>
#include <functional>
#include <random>
#include <string>
#include <sstream>
#include <vector>
#include <sys/stat.h>

// ============================================================
// [2] MATH LIBRARY
// ============================================================
struct vec2 { float x,y; };
struct vec3 {
    float x,y,z;
    vec3():x(0),y(0),z(0){}
    vec3(float a):x(a),y(a),z(a){}
    vec3(float a,float b,float c):x(a),y(b),z(c){}
    vec3 operator+(vec3 o)const{return{x+o.x,y+o.y,z+o.z};}
    vec3 operator-(vec3 o)const{return{x-o.x,y-o.y,z-o.z};}
    vec3 operator*(float s)const{return{x*s,y*s,z*s};}
    vec3 operator*(vec3 o)const{return{x*o.x,y*o.y,z*o.z};}
    vec3 operator/(float s)const{return{x/s,y/s,z/s};}
    vec3& operator+=(vec3 o){x+=o.x;y+=o.y;z+=o.z;return*this;}
    vec3& operator-=(vec3 o){x-=o.x;y-=o.y;z-=o.z;return*this;}
    vec3& operator*=(float s){x*=s;y*=s;z*=s;return*this;}
    float& operator[](int i){return(&x)[i];}
    float operator[](int i)const{return(&x)[i];}
};
vec3 operator*(float s,vec3 v){return v*s;}

struct vec4 {
    float x,y,z,w;
    vec4():x(0),y(0),z(0),w(0){}
    vec4(float a,float b,float c,float d):x(a),y(b),z(c),w(d){}
    vec4(vec3 v,float w_):x(v.x),y(v.y),z(v.z),w(w_){}
    float& operator[](int i){return(&x)[i];}
    float operator[](int i)const{return(&x)[i];}
};

struct mat4 {
    float m[16]; // column-major
    mat4(){memset(m,0,sizeof(m));}
    static mat4 identity(){mat4 r;r.m[0]=r.m[5]=r.m[10]=r.m[15]=1;return r;}
    float& operator()(int r,int c){return m[c*4+r];}
    float operator()(int r,int c)const{return m[c*4+r];}
    mat4 operator*(const mat4& b)const{
        mat4 r;
        for(int c=0;c<4;c++)for(int row=0;row<4;row++){
            float s=0;
            for(int k=0;k<4;k++) s+=m[k*4+row]*b.m[c*4+k];
            r.m[c*4+row]=s;
        }
        return r;
    }
    vec4 operator*(vec4 v)const{
        vec4 r;
        for(int i=0;i<4;i++){
            r[i]=m[0*4+i]*v.x+m[1*4+i]*v.y+m[2*4+i]*v.z+m[3*4+i]*v.w;
        }
        return r;
    }
    const float* ptr()const{return m;}
};

float dot(vec3 a,vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
vec3 cross(vec3 a,vec3 b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
float length(vec3 v){return sqrtf(dot(v,v));}
vec3 normalize(vec3 v){float l=length(v);return l>1e-8f?v/l:vec3(0);}
float clampf(float v,float lo,float hi){return v<lo?lo:v>hi?hi:v;}
float lerpf(float a,float b,float t){return a+(b-a)*t;}
vec3 lerp(vec3 a,vec3 b,float t){return a+(b-a)*t;}
float radiansf(float deg){return deg*3.14159265f/180.f;}
float smoothstep(float e0,float e1,float x){float t=clampf((x-e0)/(e1-e0),0,1);return t*t*(3-2*t);}

mat4 mat4_perspective(float fovRad,float aspect,float near,float far){
    mat4 r;
    float tanHalf=tanf(fovRad/2.f);
    r(0,0)=1.f/(aspect*tanHalf);
    r(1,1)=1.f/tanHalf;
    r(2,2)=-(far+near)/(far-near);
    r(3,2)=-1.f;
    r(2,3)=-2.f*far*near/(far-near);
    return r;
}

mat4 mat4_lookAt(vec3 eye,vec3 center,vec3 up){
    vec3 f=normalize(center-eye);
    vec3 s=normalize(cross(f,up));
    vec3 u=cross(s,f);
    mat4 r=mat4::identity();
    r(0,0)=s.x;r(0,1)=s.y;r(0,2)=s.z;
    r(1,0)=u.x;r(1,1)=u.y;r(1,2)=u.z;
    r(2,0)=-f.x;r(2,1)=-f.y;r(2,2)=-f.z;
    r(0,3)=-dot(s,eye);
    r(1,3)=-dot(u,eye);
    r(2,3)=dot(f,eye);
    return r;
}

mat4 mat4_inverse(const mat4& m){
    // cofactor expansion (Mesa-derived)
    float inv[16];
    const float* a=m.m;
    inv[0]= a[5]*(a[10]*a[15]-a[11]*a[14])-a[9]*(a[6]*a[15]-a[7]*a[14])+a[13]*(a[6]*a[11]-a[7]*a[10]);
    inv[4]=-a[4]*(a[10]*a[15]-a[11]*a[14])+a[8]*(a[6]*a[15]-a[7]*a[14])-a[12]*(a[6]*a[11]-a[7]*a[10]);
    inv[8]= a[4]*(a[9]*a[15]-a[11]*a[13])-a[8]*(a[5]*a[15]-a[7]*a[13])+a[12]*(a[5]*a[11]-a[7]*a[9]);
    inv[12]=-a[4]*(a[9]*a[14]-a[10]*a[13])+a[8]*(a[5]*a[14]-a[6]*a[13])-a[12]*(a[5]*a[10]-a[6]*a[9]);
    inv[1]=-a[1]*(a[10]*a[15]-a[11]*a[14])+a[9]*(a[2]*a[15]-a[3]*a[14])-a[13]*(a[2]*a[11]-a[3]*a[10]);
    inv[5]= a[0]*(a[10]*a[15]-a[11]*a[14])-a[8]*(a[2]*a[15]-a[3]*a[14])+a[12]*(a[2]*a[11]-a[3]*a[10]);
    inv[9]=-a[0]*(a[9]*a[15]-a[11]*a[13])+a[8]*(a[1]*a[15]-a[3]*a[13])-a[12]*(a[1]*a[11]-a[3]*a[9]);
    inv[13]= a[0]*(a[9]*a[14]-a[10]*a[13])-a[8]*(a[1]*a[14]-a[2]*a[13])+a[12]*(a[1]*a[10]-a[2]*a[9]);
    inv[2]= a[1]*(a[6]*a[15]-a[7]*a[14])-a[5]*(a[2]*a[15]-a[3]*a[14])+a[13]*(a[2]*a[7]-a[3]*a[6]);
    inv[6]=-a[0]*(a[6]*a[15]-a[7]*a[14])+a[4]*(a[2]*a[15]-a[3]*a[14])-a[12]*(a[2]*a[7]-a[3]*a[6]);
    inv[10]= a[0]*(a[5]*a[15]-a[7]*a[13])-a[4]*(a[1]*a[15]-a[3]*a[13])+a[12]*(a[1]*a[7]-a[3]*a[5]);
    inv[14]=-a[0]*(a[5]*a[14]-a[6]*a[13])+a[4]*(a[1]*a[14]-a[2]*a[13])-a[12]*(a[1]*a[6]-a[2]*a[5]);
    inv[3]=-a[1]*(a[6]*a[11]-a[7]*a[10])+a[5]*(a[2]*a[11]-a[3]*a[10])-a[9]*(a[2]*a[7]-a[3]*a[6]);
    inv[7]= a[0]*(a[6]*a[11]-a[7]*a[10])-a[4]*(a[2]*a[11]-a[3]*a[10])+a[8]*(a[2]*a[7]-a[3]*a[6]);
    inv[11]=-a[0]*(a[5]*a[11]-a[7]*a[9])+a[4]*(a[1]*a[11]-a[3]*a[9])-a[8]*(a[1]*a[7]-a[3]*a[5]);
    inv[15]= a[0]*(a[5]*a[10]-a[6]*a[9])-a[4]*(a[1]*a[10]-a[2]*a[9])+a[8]*(a[1]*a[6]-a[2]*a[5]);
    float det=a[0]*inv[0]+a[1]*inv[4]+a[2]*inv[8]+a[3]*inv[12];
    if(fabsf(det)<1e-12f) return mat4::identity();
    float invDet=1.f/det;
    mat4 r;
    for(int i=0;i<16;i++) r.m[i]=inv[i]*invDet;
    return r;
}

// ============================================================
// [3] SIMPLEX NOISE
// ============================================================
namespace simplex {
static const int perm[512]={
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};
static const float grad3[12][3]={
    {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
    {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
    {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
};
static inline int fastfloor(float x){int xi=(int)x;return x<xi?xi-1:xi;}
static inline float gdot3(const float g[3],float x,float y,float z){return g[0]*x+g[1]*y+g[2]*z;}

float noise3(float xin,float yin,float zin){
    const float F3=1.f/3.f, G3=1.f/6.f;
    float s=(xin+yin+zin)*F3;
    int i=fastfloor(xin+s),j=fastfloor(yin+s),k=fastfloor(zin+s);
    float t=(i+j+k)*G3;
    float X0=i-t,Y0=j-t,Z0=k-t;
    float x0=xin-X0,y0=yin-Y0,z0=zin-Z0;
    int i1,j1,k1,i2,j2,k2;
    if(x0>=y0){
        if(y0>=z0){i1=1;j1=0;k1=0;i2=1;j2=1;k2=0;}
        else if(x0>=z0){i1=1;j1=0;k1=0;i2=1;j2=0;k2=1;}
        else{i1=0;j1=0;k1=1;i2=1;j2=0;k2=1;}
    }else{
        if(y0<z0){i1=0;j1=0;k1=1;i2=0;j2=1;k2=1;}
        else if(x0<z0){i1=0;j1=1;k1=0;i2=0;j2=1;k2=1;}
        else{i1=0;j1=1;k1=0;i2=1;j2=1;k2=0;}
    }
    float x1=x0-i1+G3,y1=y0-j1+G3,z1=z0-k1+G3;
    float x2=x0-i2+2*G3,y2=y0-j2+2*G3,z2=z0-k2+2*G3;
    float x3=x0-1+3*G3,y3=y0-1+3*G3,z3=z0-1+3*G3;
    int ii=i&255,jj=j&255,kk=k&255;
    int gi0=perm[ii+perm[jj+perm[kk]]]%12;
    int gi1=perm[ii+i1+perm[jj+j1+perm[kk+k1]]]%12;
    int gi2=perm[ii+i2+perm[jj+j2+perm[kk+k2]]]%12;
    int gi3=perm[ii+1+perm[jj+1+perm[kk+1]]]%12;
    float n0=0,n1=0,n2=0,n3=0;
    float t0=0.6f-x0*x0-y0*y0-z0*z0;
    if(t0>=0){t0*=t0;n0=t0*t0*gdot3(grad3[gi0],x0,y0,z0);}
    float t1=0.6f-x1*x1-y1*y1-z1*z1;
    if(t1>=0){t1*=t1;n1=t1*t1*gdot3(grad3[gi1],x1,y1,z1);}
    float t2=0.6f-x2*x2-y2*y2-z2*z2;
    if(t2>=0){t2*=t2;n2=t2*t2*gdot3(grad3[gi2],x2,y2,z2);}
    float t3=0.6f-x3*x3-y3*y3-z3*z3;
    if(t3>=0){t3*=t3;n3=t3*t3*gdot3(grad3[gi3],x3,y3,z3);}
    return 32.f*(n0+n1+n2+n3);
}
float fbm3(float x,float y,float z,int oct=3){
    float v=0,amp=1,freq=1,total_amp=0;
    for(int i=0;i<oct;i++){v+=noise3(x*freq,y*freq,z*freq)*amp;total_amp+=amp;amp*=0.5f;freq*=2.f;}
    return v/total_amp;
}
} // namespace simplex

// ============================================================
// [4] CATMULL-ROM SPLINES
// ============================================================
vec3 catmull_rom_point(vec3 p0,vec3 p1,vec3 p2,vec3 p3,float t){
    float t2=t*t,t3=t2*t;
    return (p0*(-0.5f*t3+t2-0.5f*t)+p1*(1.5f*t3-2.5f*t2+1.f)+
            p2*(-1.5f*t3+2.f*t2+0.5f*t)+p3*(0.5f*t3-0.5f*t2));
}
struct SplinePoint { int x,y,z; };
std::vector<SplinePoint> rasterize_spline(vec3 p0,vec3 p1,vec3 p2,vec3 p3,int steps=20){
    std::vector<SplinePoint> pts;
    for(int i=0;i<=steps;i++){
        float t=(float)i/(float)steps;
        vec3 p=catmull_rom_point(p0,p1,p2,p3,t);
        SplinePoint sp;
        sp.x=(int)roundf(p.x); sp.y=(int)roundf(p.y); sp.z=(int)roundf(p.z);
        if(pts.empty()||pts.back().x!=sp.x||pts.back().y!=sp.y||pts.back().z!=sp.z)
            pts.push_back(sp);
    }
    return pts;
}

// ============================================================
// [5] ENUMS + CONSTANTS
// ============================================================
static const int GRID_SIZE=30;
static const int GRID_LAYERS=15;
// static const float PI=3.14159265358979f;

enum CellType : uint8_t {
    CELL_EMPTY=0,CELL_VERTICAL,CELL_HORIZONTAL,CELL_BRIDGE,CELL_FACADE,
    CELL_STAIR,CELL_PIPE,CELL_ANTENNA,CELL_CABLE,CELL_VENT,CELL_ELEVATOR,
    CELL_TYPE_COUNT
};
enum MaterialType : uint8_t {
    MAT_CONCRETE=0,MAT_GLASS,MAT_METAL,MAT_NEON,MAT_RUST,MAT_STEEL,
    MAT_COUNT
};
enum DistrictType : uint8_t {
    DIST_INDUSTRIAL=0,DIST_RESIDENTIAL,DIST_COMMERCIAL,DIST_SLUM,DIST_ELITE,
    DIST_COUNT
};
enum BiomeStratum : uint8_t {
    BIOME_UNDERGROUND=0,BIOME_SURFACE,BIOME_MIDRISE,BIOME_SKYLINE,
    BIOME_COUNT
};
enum WFCTile : uint8_t {
    WFC_EMPTY=0,WFC_FLOOR_SOLID,WFC_FLOOR_HALF_N,WFC_FLOOR_HALF_E,
    WFC_WALL_N,WFC_WALL_E,WFC_WALL_CORNER_NE,WFC_WALL_CORNER_NW,
    WFC_CORRIDOR_NS,WFC_CORRIDOR_EW,WFC_ROOM_CENTER,WFC_DOOR_N,WFC_DOOR_E,
    WFC_STAIRWELL,WFC_ELEVATOR_SHAFT,WFC_TILE_COUNT
};

static const MaterialType CELL_TO_MATERIAL[CELL_TYPE_COUNT]={
    MAT_CONCRETE,MAT_CONCRETE,MAT_CONCRETE,MAT_STEEL,MAT_GLASS,
    MAT_METAL,MAT_RUST,MAT_METAL,MAT_STEEL,MAT_METAL,MAT_GLASS
};
static const char* CELL_NAMES[CELL_TYPE_COUNT]={
    "EMPTY","VERTICAL","HORIZONTAL","BRIDGE","FACADE",
    "STAIR","PIPE","ANTENNA","CABLE","VENT","ELEVATOR"
};
static const char* DISTRICT_NAMES[DIST_COUNT]={
    "INDUSTRIAL","RESIDENTIAL","COMMERCIAL","SLUM","ELITE"
};

// ============================================================
// [6] MATERIAL + DISTRICT TABLES
// ============================================================
struct Material {
    vec3 base_color;
    float metallic,roughness,emission,alpha;
};
static const Material MATERIALS[MAT_COUNT]={
    {{0.5f,0.5f,0.6f},0.0f,0.9f,0.0f,1.0f},   // concrete
    {{0.4f,0.7f,0.9f},0.0f,0.1f,0.0f,0.3f},   // glass
    {{0.6f,0.6f,0.7f},0.8f,0.4f,0.0f,1.0f},   // metal
    {{0.1f,0.9f,0.9f},0.0f,0.2f,2.0f,1.0f},   // neon
    {{0.8f,0.4f,0.2f},0.5f,0.8f,0.0f,1.0f},   // rust
    {{0.4f,0.5f,0.6f},0.9f,0.3f,0.0f,1.0f},   // steel
};

struct DistrictProps {
    vec3 color_palette[3];
    float core_density;
    int floor_thickness;
    float vertical_variation;
    float material_weights[MAT_COUNT]; // indexed by MaterialType
    float neon_probability;
    float pipe_probability;   // L-system mod
    float elevator_probability;
};
static const DistrictProps DISTRICTS[DIST_COUNT]={
    // INDUSTRIAL
    {{{0.3f,0.3f,0.4f},{0.4f,0.5f,0.5f},{0.2f,0.3f,0.35f}},
     1.2f,2,0.3f,{0.3f,0.0f,0.4f,0.0f,0.0f,0.3f},0.1f,0.4f,0.1f},
    // RESIDENTIAL
    {{{0.6f,0.5f,0.4f},{0.7f,0.6f,0.5f},{0.5f,0.4f,0.3f}},
     0.8f,1,0.5f,{0.5f,0.3f,0.0f,0.0f,0.2f,0.0f},0.2f,0.15f,0.15f},
    // COMMERCIAL
    {{{0.2f,0.3f,0.4f},{0.3f,0.4f,0.5f},{0.1f,0.2f,0.3f}},
     0.6f,3,0.8f,{0.2f,0.6f,0.0f,0.0f,0.0f,0.2f},0.4f,0.1f,0.2f},
    // SLUM
    {{{0.4f,0.35f,0.3f},{0.5f,0.4f,0.35f},{0.45f,0.4f,0.35f}},
     1.5f,1,0.2f,{0.4f,0.0f,0.2f,0.0f,0.4f,0.0f},0.05f,0.3f,0.05f},
    // ELITE
    {{{0.8f,0.8f,0.85f},{0.75f,0.75f,0.8f},{0.7f,0.75f,0.8f}},
     0.4f,3,0.9f,{0.2f,0.5f,0.0f,0.0f,0.0f,0.3f},0.3f,0.05f,0.3f},
};

struct BiomeParams {
    int y_min,y_max;
    float density_mult;
    float decay_mult;
    float rust_mult;
};
static const BiomeParams BIOME_TABLE[BIOME_COUNT]={
    {0,2,   1.2f,0.8f,1.5f},  // underground
    {3,6,   1.0f,1.0f,1.0f},  // surface
    {7,11,  0.8f,1.2f,0.8f},  // midrise
    {12,14, 0.5f,1.5f,0.5f},  // skyline
};
static BiomeStratum get_stratum(int y){
    for(int i=0;i<BIOME_COUNT;i++) if(y>=BIOME_TABLE[i].y_min&&y<=BIOME_TABLE[i].y_max) return(BiomeStratum)i;
    return BIOME_SURFACE;
}

// ============================================================
// [7] WFC SOLVER
// ============================================================
// adjacency bitmasks: bit i means tile i is allowed as neighbor
// directions: 0=N(+z), 1=E(+x), 2=S(-z), 3=W(-x)
static uint16_t wfc_adjacency[WFC_TILE_COUNT][4];
static float wfc_base_weights[WFC_TILE_COUNT];

static void wfc_init_tables(){
    // default: all can neighbor all
    for(int i=0;i<WFC_TILE_COUNT;i++){
        for(int d=0;d<4;d++) wfc_adjacency[i][d]=0xFFFF;
        wfc_base_weights[i]=1.0f;
    }
    // empty can be next to anything
    wfc_base_weights[WFC_EMPTY]=3.0f;
    wfc_base_weights[WFC_FLOOR_SOLID]=5.0f;
    wfc_base_weights[WFC_CORRIDOR_NS]=2.0f;
    wfc_base_weights[WFC_CORRIDOR_EW]=2.0f;
    wfc_base_weights[WFC_ROOM_CENTER]=3.0f;
    wfc_base_weights[WFC_WALL_N]=1.5f;
    wfc_base_weights[WFC_WALL_E]=1.5f;
    wfc_base_weights[WFC_WALL_CORNER_NE]=0.8f;
    wfc_base_weights[WFC_WALL_CORNER_NW]=0.8f;
    wfc_base_weights[WFC_DOOR_N]=0.5f;
    wfc_base_weights[WFC_DOOR_E]=0.5f;
    wfc_base_weights[WFC_STAIRWELL]=0.3f;
    wfc_base_weights[WFC_ELEVATOR_SHAFT]=0.2f;
    wfc_base_weights[WFC_FLOOR_HALF_N]=1.0f;
    wfc_base_weights[WFC_FLOOR_HALF_E]=1.0f;
    // walls: wall_N shouldn't face another wall_N to the north
    uint16_t no_wall_n=0xFFFF & ~(1<<WFC_WALL_N);
    uint16_t no_wall_e=0xFFFF & ~(1<<WFC_WALL_E);
    wfc_adjacency[WFC_WALL_N][0]=no_wall_n; // N side
    wfc_adjacency[WFC_WALL_N][2]=no_wall_n; // S side
    wfc_adjacency[WFC_WALL_E][1]=no_wall_e;
    wfc_adjacency[WFC_WALL_E][3]=no_wall_e;
    // corridors: NS corridor connects N/S, not EW neighbors
    uint16_t corridor_ns_ew=(1<<WFC_WALL_N)|(1<<WFC_WALL_E)|(1<<WFC_WALL_CORNER_NE)|(1<<WFC_WALL_CORNER_NW)|(1<<WFC_EMPTY);
    wfc_adjacency[WFC_CORRIDOR_NS][1]=corridor_ns_ew;
    wfc_adjacency[WFC_CORRIDOR_NS][3]=corridor_ns_ew;
    wfc_adjacency[WFC_CORRIDOR_EW][0]=corridor_ns_ew;
    wfc_adjacency[WFC_CORRIDOR_EW][2]=corridor_ns_ew;
    // stairwell/elevator: must have floor or corridor neighbor
    uint16_t struct_mask=(1<<WFC_FLOOR_SOLID)|(1<<WFC_CORRIDOR_NS)|(1<<WFC_CORRIDOR_EW)|(1<<WFC_ROOM_CENTER)|(1<<WFC_DOOR_N)|(1<<WFC_DOOR_E)|(1<<WFC_STAIRWELL)|(1<<WFC_ELEVATOR_SHAFT);
    for(int d=0;d<4;d++){
        wfc_adjacency[WFC_STAIRWELL][d]=struct_mask|(1<<WFC_EMPTY);
        wfc_adjacency[WFC_ELEVATOR_SHAFT][d]=struct_mask|(1<<WFC_EMPTY);
    }
}

struct WFCCell {
    uint16_t possible; // bitmask of possible tiles
    int collapsed_tile; // -1 if not collapsed
    float entropy;
};

static float wfc_calc_entropy(uint16_t possible,const float weights[WFC_TILE_COUNT]){
    float sum=0,sumLog=0;
    for(int i=0;i<WFC_TILE_COUNT;i++){
        if(possible&(1<<i)){
            sum+=weights[i];
            sumLog+=weights[i]*logf(weights[i]+1e-6f);
        }
    }
    if(sum<1e-6f) return 0;
    return logf(sum)-sumLog/sum;
}

static int wfc_count_options(uint16_t possible){
    int c=0;for(int i=0;i<WFC_TILE_COUNT;i++) if(possible&(1<<i)) c++;return c;
}

struct WFCSolver {
    WFCCell cells[GRID_SIZE][GRID_SIZE];
    float weights[WFC_TILE_COUNT];
    std::mt19937 rng;
    int backtrack_depth;
    static const int MAX_BACKTRACK=50;
    static const int MAX_ITERATIONS=1000;

    void init(uint32_t seed,DistrictType dist,BiomeStratum stratum){
        rng.seed(seed);
        backtrack_depth=0;
        for(int i=0;i<WFC_TILE_COUNT;i++) weights[i]=wfc_base_weights[i];
        // district modifiers
        if(dist==DIST_INDUSTRIAL){weights[WFC_CORRIDOR_NS]*=1.5f;weights[WFC_CORRIDOR_EW]*=1.5f;}
        if(dist==DIST_ELITE){weights[WFC_ROOM_CENTER]*=2.f;weights[WFC_FLOOR_SOLID]*=1.5f;}
        if(dist==DIST_SLUM){weights[WFC_EMPTY]*=2.f;weights[WFC_FLOOR_HALF_N]*=2.f;}
        // stratum modifiers
        if(stratum==BIOME_UNDERGROUND){weights[WFC_CORRIDOR_NS]*=1.5f;weights[WFC_CORRIDOR_EW]*=1.5f;}
        if(stratum==BIOME_SKYLINE){weights[WFC_EMPTY]*=3.f;}
        for(int x=0;x<GRID_SIZE;x++) for(int z=0;z<GRID_SIZE;z++){
            cells[x][z].possible=(1<<WFC_TILE_COUNT)-1;
            cells[x][z].collapsed_tile=-1;
            cells[x][z].entropy=wfc_calc_entropy(cells[x][z].possible,weights);
        }
    }

    void constrain(int x,int z,WFCTile tile){
        if(x<0||x>=GRID_SIZE||z<0||z>=GRID_SIZE) return;
        cells[x][z].possible=(1<<tile);
        cells[x][z].collapsed_tile=tile;
        cells[x][z].entropy=0;
    }

    bool propagate(int sx,int sz){
        struct QE{int x,z;};
        std::vector<QE> queue;
        queue.push_back({sx,sz});
        int iters=0;
        static const int dx[4]={0,1,0,-1};
        static const int dz[4]={1,0,-1,0};
        while(!queue.empty()&&iters<MAX_ITERATIONS){
            QE q=queue.back();queue.pop_back();
            int ct=cells[q.x][q.z].collapsed_tile;
            if(ct<0) continue;
            for(int d=0;d<4;d++){
                int nx=q.x+dx[d],nz=q.z+dz[d];
                if(nx<0||nx>=GRID_SIZE||nz<0||nz>=GRID_SIZE) continue;
                if(cells[nx][nz].collapsed_tile>=0) continue;
                uint16_t allowed=wfc_adjacency[ct][d];
                uint16_t prev=cells[nx][nz].possible;
                cells[nx][nz].possible&=allowed;
                if(cells[nx][nz].possible==0) return false; // contradiction
                if(cells[nx][nz].possible!=prev){
                    cells[nx][nz].entropy=wfc_calc_entropy(cells[nx][nz].possible,weights);
                    if(wfc_count_options(cells[nx][nz].possible)==1){
                        for(int t=0;t<WFC_TILE_COUNT;t++) if(cells[nx][nz].possible&(1<<t)){cells[nx][nz].collapsed_tile=t;break;}
                        queue.push_back({nx,nz});
                    }
                }
            }
            iters++;
        }
        return true;
    }

    bool collapse_one(){
        // find lowest entropy uncollapsed cell
        float minE=1e9f;int bx=-1,bz=-1;
        for(int x=0;x<GRID_SIZE;x++) for(int z=0;z<GRID_SIZE;z++){
            if(cells[x][z].collapsed_tile>=0) continue;
            if(cells[x][z].entropy<minE){minE=cells[x][z].entropy;bx=x;bz=z;}
        }
        if(bx<0) return false; // all collapsed
        // weighted random choice
        uint16_t p=cells[bx][bz].possible;
        float total=0;
        for(int i=0;i<WFC_TILE_COUNT;i++) if(p&(1<<i)) total+=weights[i];
        std::uniform_real_distribution<float> dist(0,total);
        float r=dist(rng);
        float acc=0;
        int chosen=WFC_EMPTY;
        for(int i=0;i<WFC_TILE_COUNT;i++){
            if(!(p&(1<<i))) continue;
            acc+=weights[i];
            if(acc>=r){chosen=i;break;}
        }
        cells[bx][bz].collapsed_tile=chosen;
        cells[bx][bz].possible=(1<<chosen);
        cells[bx][bz].entropy=0;
        if(!propagate(bx,bz)){
            backtrack_depth++;
            if(backtrack_depth>MAX_BACKTRACK) return false;
            // reset this cell and try again
            cells[bx][bz].possible=(1<<WFC_TILE_COUNT)-1;
            cells[bx][bz].possible&=~(1<<chosen); // exclude failed choice
            cells[bx][bz].collapsed_tile=-1;
            cells[bx][bz].entropy=wfc_calc_entropy(cells[bx][bz].possible,weights);
            if(cells[bx][bz].possible==0){cells[bx][bz].collapsed_tile=WFC_EMPTY;cells[bx][bz].possible=1;}
            return true; // continue
        }
        return true;
    }

    void solve(){
        int iters=0;
        while(iters<MAX_ITERATIONS){
            bool more=collapse_one();
            if(!more) break;
            iters++;
        }
        // fallback: flood-fill uncollapsed cells
        for(int x=0;x<GRID_SIZE;x++) for(int z=0;z<GRID_SIZE;z++){
            if(cells[x][z].collapsed_tile<0){
                // pick most likely from remaining
                uint16_t p=cells[x][z].possible;
                float best=-1;int bt=WFC_EMPTY;
                for(int i=0;i<WFC_TILE_COUNT;i++){
                    if((p&(1<<i))&&weights[i]>best){best=weights[i];bt=i;}
                }
                cells[x][z].collapsed_tile=bt;
            }
        }
    }

    CellType tile_to_cell(WFCTile t){
        switch(t){
            case WFC_FLOOR_SOLID: case WFC_FLOOR_HALF_N: case WFC_FLOOR_HALF_E: case WFC_ROOM_CENTER: return CELL_HORIZONTAL;
            case WFC_WALL_N: case WFC_WALL_E: case WFC_WALL_CORNER_NE: case WFC_WALL_CORNER_NW: return CELL_FACADE;
            case WFC_CORRIDOR_NS: case WFC_CORRIDOR_EW: return CELL_HORIZONTAL;
            case WFC_DOOR_N: case WFC_DOOR_E: return CELL_FACADE;
            case WFC_STAIRWELL: return CELL_STAIR;
            case WFC_ELEVATOR_SHAFT: return CELL_ELEVATOR;
            default: return CELL_EMPTY;
        }
    }
};

// ============================================================
// [8] L-SYSTEM ENGINE
// ============================================================
struct LSymbol { char sym; float param; };
struct LRule { char from; std::vector<LSymbol> to; float probability; };

struct LSystem {
    std::vector<LRule> rules;
    std::mt19937 rng;

    void init_rules(DistrictType dist){
        rules.clear();
        // C = core (vertical), P = pipe, S = stair, E = elevator, B = branch, F = facade
        float pipe_p=DISTRICTS[dist].pipe_probability;
        float elev_p=DISTRICTS[dist].elevator_probability;
        float term_p=(dist==DIST_SLUM)?0.3f:0.05f;
        // rule 1: C -> C ^ C (grow up)
        rules.push_back({'C',{{'C',0},{'U',1},{'C',0}},0.6f});
        // rule 2: C -> C [+P] (branch pipe right)
        rules.push_back({'C',{{'C',0},{'[',0},{'+',0},{'P',0},{']',0}},pipe_p});
        // rule 3: C -> C [-P] (branch pipe left)
        rules.push_back({'C',{{'C',0},{'[',0},{'-',0},{'P',0},{']',0}},pipe_p});
        // rule 4: C -> C [+E] (branch elevator)
        rules.push_back({'C',{{'C',0},{'[',0},{'+',0},{'E',0},{']',0}},elev_p});
        // rule 5: C -> C S (add stair)
        rules.push_back({'C',{{'C',0},{'S',0}},0.15f});
        // rule 6: C -> (terminate early for slums)
        rules.push_back({'C',{},term_p});
    }

    std::vector<LSymbol> produce(const std::vector<LSymbol>& input,int iterations){
        std::vector<LSymbol> current=input;
        for(int iter=0;iter<iterations;iter++){
            std::vector<LSymbol> next;
            for(auto& sym:current){
                bool matched=false;
                // collect applicable rules
                std::vector<int> applicable;
                for(int i=0;i<(int)rules.size();i++){
                    if(rules[i].from==sym.sym) applicable.push_back(i);
                }
                if(!applicable.empty()){
                    // weighted random selection
                    float total=0;
                    for(int i:applicable) total+=rules[i].probability;
                    std::uniform_real_distribution<float> d(0,total);
                    float r=d(rng);
                    float acc=0;
                    for(int i:applicable){
                        acc+=rules[i].probability;
                        if(acc>=r){
                            for(auto& s:rules[i].to) next.push_back(s);
                            matched=true;
                            break;
                        }
                    }
                }
                if(!matched) next.push_back(sym);
            }
            current=next;
            if(current.size()>500) break; // safety limit
        }
        return current;
    }
};

struct TurtleState { int x,y,z; int dx,dz; }; // direction: dx,dz for horizontal, y always up for U

struct LSystemInterpreter {
    uint8_t* grid;      // pointer to generator grid
    bool* support_map;
    int size,layers;

    void interpret(const std::vector<LSymbol>& symbols,int startX,int startZ,int startY){
        TurtleState state={startX,startY,startZ,1,0};
        std::vector<TurtleState> stack;
        for(auto& sym:symbols){
            switch(sym.sym){
                case 'C': // place core
                    if(state.x>=0&&state.x<size&&state.z>=0&&state.z<size&&state.y>=0&&state.y<layers){
                        int idx=state.x*size*layers+state.z*layers+state.y;
                        grid[idx]=CELL_VERTICAL;
                        support_map[idx]=true;
                    }
                    break;
                case 'U': // move up
                    state.y+=1;
                    break;
                case 'P': // place pipe
                    for(int i=0;i<3;i++){
                        int nx=state.x+state.dx*i,nz=state.z+state.dz*i;
                        if(nx>=0&&nx<size&&nz>=0&&nz<size&&state.y>=0&&state.y<layers){
                            int idx=nx*size*layers+nz*layers+state.y;
                            if(grid[idx]==CELL_EMPTY) grid[idx]=CELL_PIPE;
                        }
                    }
                    break;
                case 'E': // place elevator
                    for(int dy=0;dy<std::min(3,layers-state.y);dy++){
                        int nx=state.x+state.dx,nz=state.z+state.dz;
                        if(nx>=0&&nx<size&&nz>=0&&nz<size){
                            int idx=nx*size*layers+nz*layers+(state.y+dy);
                            if(grid[idx]==CELL_EMPTY) grid[idx]=CELL_ELEVATOR;
                        }
                    }
                    break;
                case 'S': // place stair
                    if(state.x>=0&&state.x<size&&state.z>=0&&state.z<size&&state.y>=0&&state.y<layers){
                        int idx=state.x*size*layers+state.z*layers+state.y;
                        if(grid[idx]==CELL_VERTICAL) grid[idx]=CELL_STAIR;
                    }
                    break;
                case '+': // turn right
                    {int tmp=state.dx;state.dx=-state.dz;state.dz=tmp;}
                    break;
                case '-': // turn left
                    {int tmp=state.dx;state.dx=state.dz;state.dz=-tmp;}
                    break;
                case '[': stack.push_back(state); break;
                case ']': if(!stack.empty()){state=stack.back();stack.pop_back();} break;
                default: break;
            }
        }
    }
};

// ============================================================
// [9] BIOME LAYER DEFS (helpers used by generator)
// ============================================================
// used by generator phases
[[maybe_unused]] static float biome_density_at(int y,DistrictType dist){
    BiomeStratum s=get_stratum(y);
    return BIOME_TABLE[s].density_mult*DISTRICTS[dist].core_density;
}
[[maybe_unused]] static float biome_decay_at(int y){
    return BIOME_TABLE[get_stratum(y)].decay_mult;
}
static float biome_rust_at(int y){
    return BIOME_TABLE[get_stratum(y)].rust_mult;
}

// ============================================================
// [10] MEGASTRUCTURE GENERATOR
// ============================================================
struct MegaStructureGenerator {
    uint8_t grid[GRID_SIZE][GRID_SIZE][GRID_LAYERS];
    uint8_t district_map[GRID_SIZE][GRID_SIZE];
    bool support_map[GRID_SIZE][GRID_SIZE][GRID_LAYERS];
    int size,layers;
    std::string seed;
    std::mt19937 rng;

    void init(const std::string& s){
        seed=s;
        size=GRID_SIZE;layers=GRID_LAYERS;
        memset(grid,CELL_EMPTY,sizeof(grid));
        memset(support_map,0,sizeof(support_map));
        // seed RNG from string hash
        uint32_t h=0;
        for(char c:seed) h=h*31+c;
        rng.seed(h);
        generate_district_map();
    }

    DistrictType get_district(int x,int z){
        if(x>=0&&x<size&&z>=0&&z<size) return(DistrictType)district_map[x][z];
        return DIST_RESIDENTIAL;
    }

    void generate_district_map(){
        for(int x=0;x<size;x++) for(int z=0;z<size;z++){
            float n=simplex::noise3(x*0.05f,z*0.05f,0)*1.0f+
                     simplex::noise3(x*0.1f,z*0.1f,1)*0.5f+
                     simplex::noise3(x*0.2f,z*0.2f,2)*0.25f;
            if(n<-0.3f) district_map[x][z]=DIST_SLUM;
            else if(n<-0.1f) district_map[x][z]=DIST_INDUSTRIAL;
            else if(n<0.1f) district_map[x][z]=DIST_RESIDENTIAL;
            else if(n<0.3f) district_map[x][z]=DIST_COMMERCIAL;
            else district_map[x][z]=DIST_ELITE;
        }
    }

    void generate(){
        phase1_skeleton();
        phase2_floorplans();
        phase3_infrastructure();
        phase4_erosion();
        ensure_structural_integrity();
        add_support_pillars();
    }

    // --- phase 1: L-system skeleton ---
    void phase1_skeleton(){
        LSystem lsys;
        for(int x=0;x<size;x++) for(int z=0;z<size;z++){
            DistrictType dist=get_district(x,z);
            float base_prob=0.15f*DISTRICTS[dist].core_density;
            float noise_mod=simplex::noise3(x*0.1f,z*0.1f,3.0f)*0.1f;
            std::uniform_real_distribution<float> d01(0,1);
            if(d01(rng)<base_prob+noise_mod){
                // determine height
                float vv=DISTRICTS[dist].vertical_variation;
                int height_range=(int)(layers*vv);
                int min_h=std::max(5,layers-height_range);
                std::uniform_int_distribution<int> hd(min_h,layers-2);
                int height=hd(rng);
                // build core via L-system
                lsys.rng=rng; // share state
                lsys.init_rules(dist);
                std::vector<LSymbol> axiom;
                for(int i=0;i<height;i++){axiom.push_back({'C',0});axiom.push_back({'U',1});}
                auto result=lsys.produce(axiom,3);
                LSystemInterpreter interp;
                interp.grid=&grid[0][0][0];
                interp.support_map=&support_map[0][0][0];
                interp.size=size;interp.layers=layers;
                interp.interpret(result,x,z,0);
                rng=lsys.rng; // sync back
                // also build base core for structural support
                int base_w=(dist==DIST_SLUM)?1:2;
                for(int y=0;y<height;y++){
                    int cw=std::max(1,base_w-(int)(y/5));
                    for(int dx=-cw;dx<=cw;dx++) for(int dz=-cw;dz<=cw;dz++){
                        int nx=x+dx,nz=z+dz;
                        if(nx>=0&&nx<size&&nz>=0&&nz<size){
                            grid[nx][nz][y]=CELL_VERTICAL;
                            support_map[nx][nz][y]=true;
                        }
                    }
                }
            }
        }
    }

    // --- phase 2: WFC floor plans ---
    void phase2_floorplans(){
        wfc_init_tables();
        for(int y=0;y<layers;y++){
            // determine dominant district for this layer (center of grid)
            DistrictType dist=get_district(size/2,size/2);
            BiomeStratum stratum=get_stratum(y);
            WFCSolver wfc;
            uint32_t layer_seed=(uint32_t)(rng())^(uint32_t)(y*12345);
            wfc.init(layer_seed,dist,stratum);
            // pre-constrain cells that have L-system cores
            for(int x=0;x<size;x++) for(int z=0;z<size;z++){
                if(grid[x][z][y]==CELL_VERTICAL) wfc.constrain(x,z,WFC_STAIRWELL);
                else if(grid[x][z][y]==CELL_ELEVATOR) wfc.constrain(x,z,WFC_ELEVATOR_SHAFT);
                else if(grid[x][z][y]==CELL_STAIR) wfc.constrain(x,z,WFC_STAIRWELL);
            }
            wfc.solve();
            // write WFC results to grid (only where grid is empty or horizontal)
            for(int x=0;x<size;x++) for(int z=0;z<size;z++){
                CellType existing=(CellType)grid[x][z][y];
                if(existing==CELL_EMPTY||existing==CELL_HORIZONTAL){
                    WFCTile t=(WFCTile)wfc.cells[x][z].collapsed_tile;
                    CellType c=wfc.tile_to_cell(t);
                    if(c!=CELL_EMPTY){
                        // only place if adjacent to existing structure
                        bool adj=false;
                        if(y>0&&support_map[x][z][y-1]) adj=true;
                        for(int dd=0;dd<4&&!adj;dd++){
                            static const int ddx[4]={0,1,0,-1};
                            static const int ddz[4]={1,0,-1,0};
                            int ax=x+ddx[dd],az=z+ddz[dd];
                            if(ax>=0&&ax<size&&az>=0&&az<size&&grid[ax][az][y]!=CELL_EMPTY) adj=true;
                        }
                        if(adj){
                            grid[x][z][y]=(uint8_t)c;
                            support_map[x][z][y]=true;
                        }
                    }
                }
            }
        }
    }

    // --- phase 3: infrastructure (splines) ---
    void phase3_infrastructure(){
        add_spline_bridges();
        add_spline_cables();
        add_spline_pipes();
        add_rooftop_details();
        add_external_elevators();
    }

    void add_spline_bridges(){
        std::uniform_int_distribution<int> dy(3,layers-2);
        std::uniform_real_distribution<float> d01(0,1);
        for(int iter=0;iter<(int)(size*layers*0.02f);iter++){
            int y=dy(rng);
            // find two cores at this height
            std::vector<std::pair<int,int>> cores;
            for(int x=0;x<size;x++) for(int z=0;z<size;z++){
                if(grid[x][z][y]==CELL_VERTICAL) cores.push_back({x,z});
            }
            if(cores.size()<2) continue;
            std::uniform_int_distribution<int> ci(0,(int)cores.size()-1);
            auto[sx,sz]=cores[ci(rng)];
            auto[ex,ez]=cores[ci(rng)];
            if(sx==ex&&sz==ez) continue;
            // catmull-rom bridge with slight arc
            float midY=(float)y+1.5f; // arc up
            vec3 p0((float)sx,(float)y,(float)sz);
            vec3 p1((float)sx,(float)y,(float)sz);
            vec3 p2((float)ex,midY,(float)ez);
            vec3 p3((float)ex,(float)y,(float)ez);
            auto pts=rasterize_spline(p0,p1,p2,p3,30);
            for(auto& pt:pts){
                if(pt.x>=0&&pt.x<size&&pt.z>=0&&pt.z<size&&pt.y>=0&&pt.y<layers){
                    if(grid[pt.x][pt.z][pt.y]==CELL_EMPTY){
                        grid[pt.x][pt.z][pt.y]=CELL_BRIDGE;
                        support_map[pt.x][pt.z][pt.y]=true;
                    }
                }
            }
        }
    }

    void add_spline_cables(){
        std::uniform_real_distribution<float> d01(0,1);
        for(int iter=0;iter<(int)(size*0.5f);iter++){
            std::vector<std::tuple<int,int,int>> verts;
            for(int x=0;x<size;x++) for(int z=0;z<size;z++) for(int y=0;y<layers;y++){
                if(grid[x][z][y]==CELL_VERTICAL) verts.push_back({x,z,y});
            }
            if(verts.size()<2) continue;
            std::uniform_int_distribution<int> vi(0,(int)verts.size()-1);
            auto[sx,sz,sy]=verts[vi(rng)];
            auto[ex,ez,ey]=verts[vi(rng)];
            if(abs(sx-ex)+abs(sz-ez)>15||abs(sx-ex)+abs(sz-ez)<3) continue;
            float droop=std::min(sy,ey)-2.0f; // droop down
            vec3 p0((float)sx,(float)sy,(float)sz);
            vec3 p1((float)(sx+ex)/2,droop,(float)(sz+ez)/2);
            vec3 p2((float)(sx+ex)/2,droop+0.5f,(float)(sz+ez)/2);
            vec3 p3((float)ex,(float)ey,(float)ez);
            auto pts=rasterize_spline(p0,p1,p2,p3,25);
            for(auto& pt:pts){
                if(pt.x>=0&&pt.x<size&&pt.z>=0&&pt.z<size&&pt.y>=0&&pt.y<layers){
                    if(grid[pt.x][pt.z][pt.y]==CELL_EMPTY) grid[pt.x][pt.z][pt.y]=CELL_CABLE;
                }
            }
        }
    }

    void add_spline_pipes(){
        std::uniform_int_distribution<int> rx(0,size-1),ry(1,layers-2);
        std::uniform_real_distribution<float> d01(0,1);
        for(int iter=0;iter<(int)(size*layers*0.03f);iter++){
            int x=rx(rng),z=rx(rng),y=ry(rng);
            if(grid[x][z][y]!=CELL_VERTICAL&&grid[x][z][y]!=CELL_HORIZONTAL) continue;
            int dir=(int)(d01(rng)*4);
            static const int ddx[4]={1,-1,0,0};
            static const int ddz[4]={0,0,1,-1};
            // multi-segment spline pipe
            std::vector<vec3> waypoints;
            waypoints.push_back(vec3((float)x,(float)y,(float)z));
            int cx=x,cz=z;
            for(int seg=0;seg<3;seg++){
                cx+=ddx[dir]*(3+(int)(d01(rng)*3));
                cz+=ddz[dir]*(int)(d01(rng)*2-1);
                cx=std::clamp(cx,0,size-1);
                cz=std::clamp(cz,0,size-1);
                waypoints.push_back(vec3((float)cx,(float)y,(float)cz));
            }
            if(waypoints.size()>=4){
                auto pts=rasterize_spline(waypoints[0],waypoints[1],waypoints[2],waypoints[3],20);
                for(auto& pt:pts){
                    if(pt.x>=0&&pt.x<size&&pt.z>=0&&pt.z<size&&pt.y>=0&&pt.y<layers){
                        if(grid[pt.x][pt.z][pt.y]==CELL_EMPTY) grid[pt.x][pt.z][pt.y]=CELL_PIPE;
                    }
                }
            }
        }
    }

    void add_rooftop_details(){
        std::uniform_real_distribution<float> d01(0,1);
        std::uniform_int_distribution<int> hd(1,3);
        for(int x=0;x<size;x++) for(int z=0;z<size;z++){
            for(int y=layers-1;y>=0;y--){
                if(grid[x][z][y]!=CELL_EMPTY){
                    if(d01(rng)<0.15f&&y<layers-1){
                        CellType detail=(d01(rng)<0.5f)?CELL_ANTENNA:CELL_VENT;
                        int h=hd(rng);
                        for(int dy=1;dy<=h&&y+dy<layers;dy++) grid[x][z][y+dy]=(uint8_t)detail;
                    }
                    break;
                }
            }
        }
    }

    void add_external_elevators(){
        std::uniform_real_distribution<float> d01(0,1);
        for(int x=0;x<size;x++) for(int z=0;z<size;z++){
            bool is_core=false;
            for(int y=0;y<layers;y++) if(grid[x][z][y]==CELL_VERTICAL){is_core=true;break;}
            if(!is_core||d01(rng)>0.2f) continue;
            static const int ddx[4]={1,-1,0,0};
            static const int ddz[4]={0,0,1,-1};
            for(int d=0;d<4;d++){
                int nx=x+ddx[d],nz=z+ddz[d];
                if(nx<0||nx>=size||nz<0||nz>=size) continue;
                for(int y=0;y<layers;y++){
                    if(grid[nx][nz][y]==CELL_EMPTY){
                        if(y==0||grid[nx][nz][y-1]==CELL_ELEVATOR||grid[nx][nz][y-1]==CELL_HORIZONTAL||grid[nx][nz][y-1]==CELL_VERTICAL)
                            grid[nx][nz][y]=CELL_ELEVATOR;
                    }
                }
                break;
            }
        }
    }

    // --- phase 4: erosion ---
    void phase4_erosion(){
        erosion_structural_weakening();
        erosion_rust_spread();
        erosion_collapse_propagation();
        erosion_patina();
    }

    int count_empty_neighbors(int x,int z,int y){
        int c=0;
        static const int dx[6]={-1,1,0,0,0,0};
        static const int dy[6]={0,0,-1,1,0,0};
        static const int dz[6]={0,0,0,0,-1,1};
        for(int i=0;i<6;i++){
            int nx=x+dx[i],nz=z+dz[i],ny=y+dy[i];
            if(nx<0||nx>=size||nz<0||nz>=size||ny<0||ny>=layers){c++;continue;}
            if(grid[nx][nz][ny]==CELL_EMPTY) c++;
        }
        return c;
    }

    void erosion_structural_weakening(){
        std::uniform_real_distribution<float> d01(0,1);
        for(int x=0;x<size;x++) for(int z=0;z<size;z++) for(int y=0;y<layers;y++){
            if(grid[x][z][y]==CELL_EMPTY) continue;
            int exposure=count_empty_neighbors(x,z,y);
            float noise_t=simplex::noise3(x*0.2f,y*0.2f,z*0.2f)*0.5f+0.5f;
            BiomeStratum stratum=get_stratum(y);
            float threshold=(stratum==BIOME_SKYLINE)?0.3f:0.6f;
            if(exposure>=4&&noise_t>threshold){
                grid[x][z][y]=CELL_EMPTY;
                support_map[x][z][y]=false;
            }
        }
    }

    void erosion_rust_spread(){
        std::uniform_real_distribution<float> d01(0,1);
        // mark cells for rust (don't modify while iterating)
        std::vector<std::tuple<int,int,int>> to_rust;
        for(int x=0;x<size;x++) for(int z=0;z<size;z++) for(int y=0;y<layers;y++){
            if(grid[x][z][y]==CELL_EMPTY) continue;
            // check if neighbor has pipe/rust
            bool adj_rust=false;
            static const int ddx[6]={-1,1,0,0,0,0};
            static const int ddy[6]={0,0,-1,1,0,0};
            static const int ddz[6]={0,0,0,0,-1,1};
            for(int i=0;i<6;i++){
                int nx=x+ddx[i],nz=z+ddz[i],ny=y+ddy[i];
                if(nx>=0&&nx<size&&nz>=0&&nz<size&&ny>=0&&ny<layers){
                    if(grid[nx][nz][ny]==CELL_PIPE) adj_rust=true;
                }
            }
            if(adj_rust){
                float prob=0.15f*biome_rust_at(y);
                if(d01(rng)<prob) to_rust.push_back({x,z,y});
            }
        }
        // apply: just mark material as rust (we'll handle in rendering via noise)
        // for generation, convert some cells to PIPE (visual rust indicator)
        for(auto&[x,z,y]:to_rust){
            CellType ct=(CellType)grid[x][z][y];
            if(ct==CELL_HORIZONTAL||ct==CELL_FACADE||ct==CELL_VERTICAL){
                // keep structure but will tint rust in renderer
            }
        }
    }

    void erosion_collapse_propagation(){
        // unsupported cells above removed cells cascade (max 3 up)
        for(int pass=0;pass<3;pass++){
            for(int x=0;x<size;x++) for(int z=0;z<size;z++){
                for(int y=1;y<layers;y++){
                    if(grid[x][z][y]!=CELL_EMPTY&&grid[x][z][y]!=CELL_VERTICAL){
                        if(!has_support(x,z,y)){
                            grid[x][z][y]=CELL_EMPTY;
                            support_map[x][z][y]=false;
                        }
                    }
                }
            }
        }
    }

    void erosion_patina(){
        // handled in renderer via simplex noise color darkening
    }

    bool has_support(int x,int z,int y){
        if(y<=0) return true;
        if(support_map[x][z][y-1]) return true;
        static const int ddx[4]={-1,1,0,0};
        static const int ddz[4]={0,0,-1,1};
        for(int d=0;d<4;d++){
            int nx=x+ddx[d],nz=z+ddz[d];
            if(nx>=0&&nx<size&&nz>=0&&nz<size){
                if(grid[nx][nz][y]==CELL_HORIZONTAL||grid[nx][nz][y]==CELL_BRIDGE) return true;
            }
        }
        return false;
    }

    void ensure_structural_integrity(){
        for(int y=1;y<layers;y++) for(int x=0;x<size;x++) for(int z=0;z<size;z++){
            CellType ct=(CellType)grid[x][z][y];
            if(ct==CELL_HORIZONTAL||ct==CELL_FACADE){
                if(!has_support(x,z,y)){
                    grid[x][z][y]=CELL_EMPTY;
                    support_map[x][z][y]=false;
                }
            }
        }
    }

    void add_support_pillars(){
        for(int x=0;x<size;x++) for(int z=0;z<size;z++) for(int y=1;y<layers;y++){
            if(grid[x][z][y]==CELL_HORIZONTAL&&!has_support(x,z,y)){
                for(int py=y-1;py>=0;py--){
                    if(grid[x][z][py]==CELL_EMPTY){
                        grid[x][z][py]=CELL_VERTICAL;
                        support_map[x][z][py]=true;
                    } else break;
                }
            }
        }
    }

    // random missing sections
    void add_random_missing_sections(){
        std::uniform_int_distribution<int> rx(0,size-1),ry(2,layers-2),rr(1,2);
        for(int iter=0;iter<(int)(size*layers*0.02f);iter++){
            int x=rx(rng),z=rx(rng),y=ry(rng),r=rr(rng);
            for(int dx=-r;dx<=r;dx++) for(int dz=-r;dz<=r;dz++){
                int nx=x+dx,nz=z+dz;
                if(nx>=0&&nx<size&&nz>=0&&nz<size){
                    CellType ct=(CellType)grid[nx][nz][y];
                    if(ct!=CELL_VERTICAL&&ct!=CELL_EMPTY){
                        grid[nx][nz][y]=CELL_EMPTY;
                        support_map[nx][nz][y]=false;
                    }
                }
            }
        }
    }
};

// ============================================================
// [11] ORBITAL CAMERA
// ============================================================
struct OrbitalCamera {
    vec3 target;
    float distance,angle,pitch;
    float target_angle,target_pitch,target_distance;
    float angle_velocity,pitch_velocity,zoom_velocity;
    float damping;
    float min_distance,max_distance;
    vec3 position;

    void init(vec3 tgt,float dist,float a=45.f,float p=30.f){
        target=tgt;distance=dist;angle=a;pitch=p;
        target_angle=a;target_pitch=p;target_distance=dist;
        angle_velocity=pitch_velocity=zoom_velocity=0;
        damping=0.85f;
        min_distance=dist*0.3f;max_distance=dist*5.f;
        position=calc_position();
    }
    vec3 calc_position(){
        float ra=radiansf(angle),rp=radiansf(pitch);
        float x=distance*cosf(rp)*cosf(ra);
        float y=distance*sinf(rp);
        float z=distance*cosf(rp)*sinf(ra);
        return target+vec3(x,y,z);
    }
    void update(float dt){
        float ad=target_angle-angle,pd=target_pitch-pitch,zd=target_distance-distance;
        angle_velocity+=ad*dt*5.f; pitch_velocity+=pd*dt*5.f; zoom_velocity+=zd*dt*3.f;
        angle_velocity*=damping; pitch_velocity*=damping; zoom_velocity*=damping;
        angle+=angle_velocity*dt; pitch+=pitch_velocity*dt; distance+=zoom_velocity*dt;
        pitch=clampf(pitch,-89.f,89.f);
        distance=clampf(distance,min_distance,max_distance);
        target_distance=clampf(target_distance,min_distance,max_distance);
        position=calc_position();
    }
    mat4 get_view(){return mat4_lookAt(position,target,vec3(0,1,0));}
    void rotate(float da,float dp=0){target_angle+=da;target_pitch+=dp;}
    void zoom(float d){
        float mn=std::max(target.x,std::max(target.y,target.z))*0.3f;
        float mx=std::max(target.x,std::max(target.y,target.z))*4.f;
        target_distance=clampf(target_distance+d,mn,mx);
    }
    void pan(float dx,float dy){
        vec3 fwd=normalize(target-position);
        vec3 right=normalize(cross(fwd,vec3(0,1,0)));
        vec3 up=cross(right,fwd);
        target+=right*(dx*0.1f)+up*(dy*0.1f);
    }
    void set_preset(int p){
        static const float presets[5][2]={{0,89},{0,0},{90,0},{45,30},{45,35.264f}};
        if(p>=0&&p<5){target_angle=presets[p][0];target_pitch=presets[p][1];angle_velocity=pitch_velocity=0;}
    }
};

// ============================================================
// [12] FPS CAMERA + COLLISION
// ============================================================
struct FPSCamera {
    vec3 position;
    float yaw,pitch_angle;
    float speed,sensitivity;
    float vy; // vertical velocity
    bool on_ground;
    static constexpr float EYE_HEIGHT=1.6f;
    static constexpr float RADIUS=0.3f;
    static constexpr float GRAVITY=9.8f;
    static constexpr float JUMP_VEL=5.0f;
    static constexpr float MAX_DELTA=0.5f;

    void init(vec3 pos){
        position=pos;yaw=-90.f;pitch_angle=0;
        speed=5.f;sensitivity=0.1f;vy=0;on_ground=false;
    }
    vec3 get_front(){
        float ry=radiansf(yaw),rp=radiansf(pitch_angle);
        return normalize(vec3(cosf(ry)*cosf(rp),sinf(rp),sinf(ry)*cosf(rp)));
    }
    mat4 get_view(){
        vec3 front=get_front();
        vec3 eye=position+vec3(0,EYE_HEIGHT,0);
        return mat4_lookAt(eye,eye+front,vec3(0,1,0));
    }
    void mouse_move(float dx,float dy){
        yaw+=dx*sensitivity;
        pitch_angle-=dy*sensitivity;
        pitch_angle=clampf(pitch_angle,-89.f,89.f);
    }
    void jump(){if(on_ground){vy=JUMP_VEL;on_ground=false;}}

    bool collides_at(vec3 pos,const uint8_t grid[GRID_SIZE][GRID_SIZE][GRID_LAYERS]){
        float offsets[4][2]={{-RADIUS,-RADIUS},{RADIUS,-RADIUS},{-RADIUS,RADIUS},{RADIUS,RADIUS}};
        float heights[2]={0,EYE_HEIGHT};
        for(auto& h:heights) for(auto& o:offsets){
            int gx=(int)floorf(pos.x+o[0]);
            int gy=(int)floorf(pos.y+h);
            int gz=(int)floorf(pos.z+o[1]);
            if(gx<0||gx>=GRID_SIZE||gz<0||gz>=GRID_SIZE||gy<0||gy>=GRID_LAYERS) continue;
            if(grid[gx][gz][gy]!=CELL_EMPTY) return true;
        }
        return false;
    }

    void update(float dt,float moveF,float moveR,const uint8_t grid[GRID_SIZE][GRID_SIZE][GRID_LAYERS]){
        vec3 front=get_front();
        vec3 right=normalize(cross(front,vec3(0,1,0)));
        vec3 move=(vec3(front.x,0,front.z)*moveF+vec3(right.x,0,right.z)*moveR)*speed*dt;
        float ml=length(move);
        if(ml>MAX_DELTA) move=move*(MAX_DELTA/ml);
        vec3 newpos=position;
        newpos.x+=move.x;
        if(collides_at(newpos,grid)) newpos.x=position.x;
        newpos.z+=move.z;
        if(collides_at(newpos,grid)) newpos.z=position.z;
        vy-=GRAVITY*dt;
        float dy=vy*dt;
        if(fabsf(dy)>MAX_DELTA) dy=(dy>0?MAX_DELTA:-MAX_DELTA);
        newpos.y+=dy;
        if(collides_at(newpos,grid)){
            if(vy<0) on_ground=true;
            vy=0;
            newpos.y=position.y;
        } else on_ground=false;
        if(newpos.y<0){newpos.y=0;vy=0;on_ground=true;}
        position=newpos;
    }
};

// ============================================================
// [13] GLSL SHADERS
// ============================================================
static const char* SCENE_VERT=R"(
#version 330 core
uniform mat4 view;
uniform mat4 projection;
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec3 in_offset;
layout(location=3) in vec3 in_color;
layout(location=4) in float in_metallic;
layout(location=5) in float in_roughness;
layout(location=6) in float in_emission;
layout(location=7) in float in_alpha;
layout(location=8) in float in_ao;
out vec3 frag_pos;
out vec3 frag_normal;
out vec3 frag_color;
out float frag_metallic;
out float frag_roughness;
out float frag_emission;
out float frag_alpha;
out float frag_ao;
void main(){
    vec3 wp=in_position+in_offset;
    gl_Position=projection*view*vec4(wp,1.0);
    frag_pos=wp;
    frag_normal=in_normal;
    frag_color=in_color;
    frag_metallic=in_metallic;
    frag_roughness=in_roughness;
    frag_emission=in_emission;
    frag_alpha=in_alpha;
    frag_ao=in_ao;
}
)";
static const char* SCENE_FRAG=R"(
#version 330 core
in vec3 frag_pos;
in vec3 frag_normal;
in vec3 frag_color;
in float frag_metallic;
in float frag_roughness;
in float frag_emission;
in float frag_alpha;
in float frag_ao;
layout(location=0) out vec4 out_color;
layout(location=1) out vec4 out_worldpos;
void main(){
    vec3 N=normalize(frag_normal);
    vec3 sun_dir=normalize(vec3(0.5,1.0,0.3));
    vec3 fill_dir=normalize(vec3(-0.3,0.5,-0.7));
    vec3 V=normalize(-frag_pos);
    vec3 H=normalize(sun_dir+V);
    float NdotL=max(dot(N,sun_dir),0.0);
    float NdotH=max(dot(N,H),0.0);
    float spec=pow(NdotH,mix(8.0,64.0,1.0-frag_roughness));
    float fill_NdotL=max(dot(N,fill_dir),0.0)*0.3;
    vec3 ambient=frag_color*0.3*frag_ao;
    vec3 diffuse=frag_color*(NdotL+fill_NdotL);
    vec3 spec_color=mix(vec3(0.04),frag_color,frag_metallic);
    vec3 specular=spec_color*spec*mix(0.5,2.0,frag_metallic);
    float edge=1.0-abs(dot(N,V));
    edge=pow(edge,3.0)*0.3;
    float fresnel=pow(1.0-max(dot(N,V),0.0),3.0);
    vec3 emissive=frag_color*frag_emission*(0.5+fresnel*1.5);
    vec3 final_color=ambient+diffuse+specular+emissive;
    final_color-=vec3(edge);
    final_color=max(final_color,vec3(0.0));
    out_color=vec4(final_color,frag_alpha);
    out_worldpos=vec4(frag_pos,1.0);
}
)";
static const char* BLOOM_EXTRACT_FRAG=R"(
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D scene_texture;
uniform float bloom_threshold;
void main(){
    vec3 c=texture(scene_texture,uv).rgb;
    float lum=dot(c,vec3(0.2126,0.7152,0.0722));
    fragColor=(lum>bloom_threshold)?vec4(c,1.0):vec4(0.0,0.0,0.0,1.0);
}
)";
static const char* BLUR_FRAG=R"(
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D input_texture;
uniform vec2 blur_direction;
uniform vec2 texture_size;
const float w[5]=float[](0.227027,0.1945946,0.1216216,0.054054,0.016216);
void main(){
    vec2 off=1.0/texture_size*blur_direction;
    vec3 r=texture(input_texture,uv).rgb*w[0];
    for(int i=1;i<5;i++){
        r+=texture(input_texture,uv+off*float(i)).rgb*w[i];
        r+=texture(input_texture,uv-off*float(i)).rgb*w[i];
    }
    fragColor=vec4(r,1.0);
}
)";
static const char* COMPOSITE_FRAG=R"(
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D scene_texture;
uniform sampler2D depth_texture;
uniform sampler2D world_pos_texture;
uniform sampler2D bloom_texture;
uniform float fog_density;
uniform vec3 fog_color;
uniform float fog_height_falloff;
uniform float bloom_intensity;
uniform bool enable_postprocessing;
uniform float time;
float linearDepth(float d,float n,float f){float z=d*2.0-1.0;return(2.0*n*f)/(f+n-z*(f-n));}
vec3 aces(vec3 x){
    float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0,1.0);
}
void main(){
    vec3 color=texture(scene_texture,uv).rgb;
    if(!enable_postprocessing){fragColor=vec4(color,1.0);return;}
    float depth=texture(depth_texture,uv).r;
    vec3 wp=texture(world_pos_texture,uv).rgb;
    vec3 bloom=texture(bloom_texture,uv).rgb;
    color+=bloom*bloom_intensity;
    float ld=linearDepth(depth,0.1,500.0);
    float hf=exp(-wp.y*fog_height_falloff);
    hf=clamp(hf,0.0,1.0);
    float cd=fog_density*(1.0+hf*2.0);
    float ff=exp(-cd*ld*0.01);
    ff=clamp(ff,0.0,1.0);
    color=mix(fog_color,color,ff);
    color=aces(color);
    vec2 vc=uv-0.5;
    float vf=1.0-dot(vc,vc)*0.5;
    color*=vf;
    float ca_str=0.002;
    float r=texture(scene_texture,uv+vec2(ca_str,0)).r;
    float b2=texture(scene_texture,uv-vec2(ca_str,0)).b;
    color.r=mix(color.r,r,0.3);
    color.b=mix(color.b,b2,0.3);
    float grain=fract(sin(dot(uv*time,vec2(12.9898,78.233)))*43758.5453);
    color+=vec3((grain-0.5)*0.03);
    fragColor=vec4(color,1.0);
}
)";
static const char* QUAD_VERT=R"(
#version 330 core
layout(location=0) in vec2 in_position;
layout(location=1) in vec2 in_texcoord;
out vec2 uv;
void main(){gl_Position=vec4(in_position,0.0,1.0);uv=in_texcoord;}
)";
static const char* UI_FRAG=R"(
#version 330 core
in vec2 uv;
out vec4 fragColor;
uniform sampler2D ui_texture;
void main(){fragColor=texture(ui_texture,uv);}
)";

// ============================================================
// [14] RENDERER
// ============================================================
// 8x8 bitmap font (printable ASCII 32-126)
static const uint8_t FONT8X8[96][8]={
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // space
    {0x18,0x3C,0x3C,0x18,0x18,0x00,0x18,0x00}, // !
    {0x36,0x36,0x00,0x00,0x00,0x00,0x00,0x00}, // "
    {0x36,0x36,0x7F,0x36,0x7F,0x36,0x36,0x00}, // #
    {0x0C,0x3E,0x03,0x1E,0x30,0x1F,0x0C,0x00}, // $
    {0x00,0x63,0x33,0x18,0x0C,0x66,0x63,0x00}, // %
    {0x1C,0x36,0x1C,0x6E,0x3B,0x33,0x6E,0x00}, // &
    {0x06,0x06,0x03,0x00,0x00,0x00,0x00,0x00}, // '
    {0x18,0x0C,0x06,0x06,0x06,0x0C,0x18,0x00}, // (
    {0x06,0x0C,0x18,0x18,0x18,0x0C,0x06,0x00}, // )
    {0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00}, // *
    {0x00,0x0C,0x0C,0x3F,0x0C,0x0C,0x00,0x00}, // +
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x06}, // ,
    {0x00,0x00,0x00,0x3F,0x00,0x00,0x00,0x00}, // -
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x00}, // .
    {0x60,0x30,0x18,0x0C,0x06,0x03,0x01,0x00}, // /
    {0x3E,0x63,0x73,0x7B,0x6F,0x67,0x3E,0x00}, // 0
    {0x0C,0x0E,0x0C,0x0C,0x0C,0x0C,0x3F,0x00}, // 1
    {0x1E,0x33,0x30,0x1C,0x06,0x33,0x3F,0x00}, // 2
    {0x1E,0x33,0x30,0x1C,0x30,0x33,0x1E,0x00}, // 3
    {0x38,0x3C,0x36,0x33,0x7F,0x30,0x78,0x00}, // 4
    {0x3F,0x03,0x1F,0x30,0x30,0x33,0x1E,0x00}, // 5
    {0x1C,0x06,0x03,0x1F,0x33,0x33,0x1E,0x00}, // 6
    {0x3F,0x33,0x30,0x18,0x0C,0x0C,0x0C,0x00}, // 7
    {0x1E,0x33,0x33,0x1E,0x33,0x33,0x1E,0x00}, // 8
    {0x1E,0x33,0x33,0x3E,0x30,0x18,0x0E,0x00}, // 9
    {0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x00}, // :
    {0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x06}, // ;
    {0x18,0x0C,0x06,0x03,0x06,0x0C,0x18,0x00}, // <
    {0x00,0x00,0x3F,0x00,0x00,0x3F,0x00,0x00}, // =
    {0x06,0x0C,0x18,0x30,0x18,0x0C,0x06,0x00}, // >
    {0x1E,0x33,0x30,0x18,0x0C,0x00,0x0C,0x00}, // ?
    {0x3E,0x63,0x7B,0x7B,0x7B,0x03,0x1E,0x00}, // @
    {0x0C,0x1E,0x33,0x33,0x3F,0x33,0x33,0x00}, // A
    {0x3F,0x66,0x66,0x3E,0x66,0x66,0x3F,0x00}, // B
    {0x3C,0x66,0x03,0x03,0x03,0x66,0x3C,0x00}, // C
    {0x1F,0x36,0x66,0x66,0x66,0x36,0x1F,0x00}, // D
    {0x7F,0x46,0x16,0x1E,0x16,0x46,0x7F,0x00}, // E
    {0x7F,0x46,0x16,0x1E,0x16,0x06,0x0F,0x00}, // F
    {0x3C,0x66,0x03,0x03,0x73,0x66,0x7C,0x00}, // G
    {0x33,0x33,0x33,0x3F,0x33,0x33,0x33,0x00}, // H
    {0x1E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, // I
    {0x78,0x30,0x30,0x30,0x33,0x33,0x1E,0x00}, // J
    {0x67,0x66,0x36,0x1E,0x36,0x66,0x67,0x00}, // K
    {0x0F,0x06,0x06,0x06,0x46,0x66,0x7F,0x00}, // L
    {0x63,0x77,0x7F,0x7F,0x6B,0x63,0x63,0x00}, // M
    {0x63,0x67,0x6F,0x7B,0x73,0x63,0x63,0x00}, // N
    {0x1C,0x36,0x63,0x63,0x63,0x36,0x1C,0x00}, // O
    {0x3F,0x66,0x66,0x3E,0x06,0x06,0x0F,0x00}, // P
    {0x1E,0x33,0x33,0x33,0x3B,0x1E,0x38,0x00}, // Q
    {0x3F,0x66,0x66,0x3E,0x36,0x66,0x67,0x00}, // R
    {0x1E,0x33,0x07,0x0E,0x38,0x33,0x1E,0x00}, // S
    {0x3F,0x2D,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, // T
    {0x33,0x33,0x33,0x33,0x33,0x33,0x3F,0x00}, // U
    {0x33,0x33,0x33,0x33,0x33,0x1E,0x0C,0x00}, // V
    {0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x00}, // W
    {0x63,0x63,0x36,0x1C,0x1C,0x36,0x63,0x00}, // X
    {0x33,0x33,0x33,0x1E,0x0C,0x0C,0x1E,0x00}, // Y
    {0x7F,0x63,0x31,0x18,0x4C,0x66,0x7F,0x00}, // Z
    {0x1E,0x06,0x06,0x06,0x06,0x06,0x1E,0x00}, // [
    {0x03,0x06,0x0C,0x18,0x30,0x60,0x40,0x00}, // backslash
    {0x1E,0x18,0x18,0x18,0x18,0x18,0x1E,0x00}, // ]
    {0x08,0x1C,0x36,0x63,0x00,0x00,0x00,0x00}, // ^
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF}, // _
    {0x0C,0x0C,0x18,0x00,0x00,0x00,0x00,0x00}, // `
    {0x00,0x00,0x1E,0x30,0x3E,0x33,0x6E,0x00}, // a
    {0x07,0x06,0x06,0x3E,0x66,0x66,0x3B,0x00}, // b
    {0x00,0x00,0x1E,0x33,0x03,0x33,0x1E,0x00}, // c
    {0x38,0x30,0x30,0x3E,0x33,0x33,0x6E,0x00}, // d
    {0x00,0x00,0x1E,0x33,0x3F,0x03,0x1E,0x00}, // e
    {0x1C,0x36,0x06,0x0F,0x06,0x06,0x0F,0x00}, // f
    {0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x1F}, // g
    {0x07,0x06,0x36,0x6E,0x66,0x66,0x67,0x00}, // h
    {0x0C,0x00,0x0E,0x0C,0x0C,0x0C,0x1E,0x00}, // i
    {0x30,0x00,0x30,0x30,0x30,0x33,0x33,0x1E}, // j
    {0x07,0x06,0x66,0x36,0x1E,0x36,0x67,0x00}, // k
    {0x0E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, // l
    {0x00,0x00,0x33,0x7F,0x7F,0x6B,0x63,0x00}, // m
    {0x00,0x00,0x1F,0x33,0x33,0x33,0x33,0x00}, // n
    {0x00,0x00,0x1E,0x33,0x33,0x33,0x1E,0x00}, // o
    {0x00,0x00,0x3B,0x66,0x66,0x3E,0x06,0x0F}, // p
    {0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x78}, // q
    {0x00,0x00,0x3B,0x6E,0x66,0x06,0x0F,0x00}, // r
    {0x00,0x00,0x3E,0x03,0x1E,0x30,0x1F,0x00}, // s
    {0x08,0x0C,0x3E,0x0C,0x0C,0x2C,0x18,0x00}, // t
    {0x00,0x00,0x33,0x33,0x33,0x33,0x6E,0x00}, // u
    {0x00,0x00,0x33,0x33,0x33,0x1E,0x0C,0x00}, // v
    {0x00,0x00,0x63,0x6B,0x7F,0x7F,0x36,0x00}, // w
    {0x00,0x00,0x63,0x36,0x1C,0x36,0x63,0x00}, // x
    {0x00,0x00,0x33,0x33,0x33,0x3E,0x30,0x1F}, // y
    {0x00,0x00,0x3F,0x19,0x0C,0x26,0x3F,0x00}, // z
    {0x38,0x0C,0x0C,0x07,0x0C,0x0C,0x38,0x00}, // {
    {0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x00}, // |
    {0x07,0x0C,0x0C,0x38,0x0C,0x0C,0x07,0x00}, // }
    {0x6E,0x3B,0x00,0x00,0x00,0x00,0x00,0x00}, // ~
};

// TGA screenshot writer
static bool write_tga(const char* path,int w,int h,const uint8_t* rgb){
    FILE* f=fopen(path,"wb");
    if(!f) return false;
    uint8_t hdr[18]={};
    hdr[2]=2;
    hdr[12]=(uint8_t)(w&0xFF);hdr[13]=(uint8_t)((w>>8)&0xFF);
    hdr[14]=(uint8_t)(h&0xFF);hdr[15]=(uint8_t)((h>>8)&0xFF);
    hdr[16]=24;
    fwrite(hdr,1,18,f);
    for(int y=0;y<h;y++){
        for(int x=0;x<w;x++){
            int i=(y*w+x)*3;
            uint8_t bgr[3]={rgb[i+2],rgb[i+1],rgb[i]};
            fwrite(bgr,1,3,f);
        }
    }
    fclose(f);
    return true;
}

// GL helpers
static GLuint compile_shader(GLenum type,const char* src){
    GLuint s=glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    int ok;glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){char log[512];glGetShaderInfoLog(s,512,nullptr,log);fprintf(stderr,"Shader error: %s\n",log);}
    return s;
}
static GLuint link_program(GLuint vs,GLuint fs){
    GLuint p=glCreateProgram();
    glAttachShader(p,vs);glAttachShader(p,fs);
    glLinkProgram(p);
    int ok;glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){char log[512];glGetProgramInfoLog(p,512,nullptr,log);fprintf(stderr,"Link error: %s\n",log);}
    glDeleteShader(vs);glDeleteShader(fs);
    return p;
}
static GLuint make_program(const char* vsrc,const char* fsrc){
    return link_program(compile_shader(GL_VERTEX_SHADER,vsrc),compile_shader(GL_FRAGMENT_SHADER,fsrc));
}

// cube with face normals: 6 faces x 4 verts x 6 floats (pos+normal) = 144 floats, 36 indices
static void gen_cube_geometry(std::vector<float>& verts,std::vector<uint32_t>& indices){
    struct Face{vec3 n; vec3 v[4];};
    Face faces[6]={
        {{0,0,1}, {{-0.4f,-0.4f,0.4f},{0.4f,-0.4f,0.4f},{0.4f,0.4f,0.4f},{-0.4f,0.4f,0.4f}}},
        {{0,0,-1},{{0.4f,-0.4f,-0.4f},{-0.4f,-0.4f,-0.4f},{-0.4f,0.4f,-0.4f},{0.4f,0.4f,-0.4f}}},
        {{1,0,0}, {{0.4f,-0.4f,0.4f},{0.4f,-0.4f,-0.4f},{0.4f,0.4f,-0.4f},{0.4f,0.4f,0.4f}}},
        {{-1,0,0},{{-0.4f,-0.4f,-0.4f},{-0.4f,-0.4f,0.4f},{-0.4f,0.4f,0.4f},{-0.4f,0.4f,-0.4f}}},
        {{0,1,0}, {{-0.4f,0.4f,0.4f},{0.4f,0.4f,0.4f},{0.4f,0.4f,-0.4f},{-0.4f,0.4f,-0.4f}}},
        {{0,-1,0},{{-0.4f,-0.4f,-0.4f},{0.4f,-0.4f,-0.4f},{0.4f,-0.4f,0.4f},{-0.4f,-0.4f,0.4f}}},
    };
    for(int f=0;f<6;f++){
        uint32_t base=(uint32_t)(verts.size()/6);
        for(int v=0;v<4;v++){
            verts.push_back(faces[f].v[v].x);verts.push_back(faces[f].v[v].y);verts.push_back(faces[f].v[v].z);
            verts.push_back(faces[f].n.x);verts.push_back(faces[f].n.y);verts.push_back(faces[f].n.z);
        }
        indices.push_back(base);indices.push_back(base+1);indices.push_back(base+2);
        indices.push_back(base);indices.push_back(base+2);indices.push_back(base+3);
    }
}

// ============================================================
// GLOBAL STATE
// ============================================================
struct AppState {
    GLFWwindow* window;
    int win_w,win_h;
    MegaStructureGenerator gen;
    OrbitalCamera orbital;
    FPSCamera fps;
    bool fps_mode;
    GLuint scene_prog,bloom_extract_prog,blur_prog,composite_prog,ui_prog;
    GLuint cube_vao,cube_vbo,cube_ibo;
    int cube_index_count;
    GLuint inst_offset_vbo,inst_color_vbo,inst_metallic_vbo,inst_roughness_vbo;
    GLuint inst_emission_vbo,inst_alpha_vbo,inst_ao_vbo;
    int instance_count;
    GLuint scene_fbo,scene_color_tex,scene_worldpos_tex,scene_depth_tex;
    GLuint bloom_extract_fbo,bloom_extract_tex;
    GLuint blur_fbo[2],blur_tex[2];
    GLuint quad_vao,quad_vbo;
    GLuint font_tex;
    float fog_density,fog_height_falloff;
    float bloom_threshold,bloom_intensity;
    int blur_iterations;
    bool enable_postprocessing;
    bool inspection_mode;
    bool show_legend;
    bool mouse_dragging;
    double last_mx,last_my;
    bool keys[512];
    float time_val;
    int sel_x,sel_y,sel_z;
    bool has_selection;
    std::string seed;
} g;

// ============================================================
// RENDERER FUNCTIONS
// ============================================================
static void create_fbos(){
    int w=g.win_w,h=g.win_h;
    // scene FBO with depth texture (not RBO, so composite can sample it)
    glGenFramebuffers(1,&g.scene_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER,g.scene_fbo);
    glGenTextures(1,&g.scene_color_tex);
    glBindTexture(GL_TEXTURE_2D,g.scene_color_tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,w,h,0,GL_RGBA,GL_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,g.scene_color_tex,0);
    glGenTextures(1,&g.scene_worldpos_tex);
    glBindTexture(GL_TEXTURE_2D,g.scene_worldpos_tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,w,h,0,GL_RGBA,GL_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,g.scene_worldpos_tex,0);
    GLenum bufs[2]={GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2,bufs);
    // depth as texture so composite shader can sample
    glGenTextures(1,&g.scene_depth_tex);
    glBindTexture(GL_TEXTURE_2D,g.scene_depth_tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT24,w,h,0,GL_DEPTH_COMPONENT,GL_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,g.scene_depth_tex,0);
    // bloom extract FBO
    glGenFramebuffers(1,&g.bloom_extract_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER,g.bloom_extract_fbo);
    glGenTextures(1,&g.bloom_extract_tex);
    glBindTexture(GL_TEXTURE_2D,g.bloom_extract_tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,w,h,0,GL_RGBA,GL_FLOAT,nullptr);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,g.bloom_extract_tex,0);
    // blur ping-pong
    for(int i=0;i<2;i++){
        glGenFramebuffers(1,&g.blur_fbo[i]);
        glBindFramebuffer(GL_FRAMEBUFFER,g.blur_fbo[i]);
        glGenTextures(1,&g.blur_tex[i]);
        glBindTexture(GL_TEXTURE_2D,g.blur_tex[i]);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,w,h,0,GL_RGBA,GL_FLOAT,nullptr);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,g.blur_tex[i],0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER,0);
}

static void create_quad(){
    float qv[]={-1,1,0,1, -1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,-1,1,0, 1,1,1,1};
    glGenVertexArrays(1,&g.quad_vao);
    glGenBuffers(1,&g.quad_vbo);
    glBindVertexArray(g.quad_vao);
    glBindBuffer(GL_ARRAY_BUFFER,g.quad_vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(qv),qv,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,16,(void*)0);
    glEnableVertexAttribArray(1);glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,16,(void*)8);
    glBindVertexArray(0);
}

static void create_font_texture(){
    int tw=128,th=48;
    std::vector<uint8_t> pixels(tw*th*4,0);
    for(int ci=0;ci<96;ci++){
        int gx=(ci%16)*8, gy=(ci/16)*8;
        for(int row=0;row<8;row++) for(int col=0;col<8;col++){
            bool on=(FONT8X8[ci][row]>>(7-col))&1;
            int px=gx+col, py=gy+row;
            int idx=(py*tw+px)*4;
            pixels[idx]=pixels[idx+1]=pixels[idx+2]=on?255:0;
            pixels[idx+3]=on?255:0;
        }
    }
    glGenTextures(1,&g.font_tex);
    glBindTexture(GL_TEXTURE_2D,g.font_tex);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,tw,th,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
}

static float compute_ao(int x,int z,int y,const MegaStructureGenerator& gen){
    int occupied=0;
    static const int dx[6]={-1,1,0,0,0,0};
    static const int dy[6]={0,0,-1,1,0,0};
    static const int dz[6]={0,0,0,0,-1,1};
    for(int i=0;i<6;i++){
        int nx=x+dx[i],nz=z+dz[i],ny=y+dy[i];
        if(nx>=0&&nx<GRID_SIZE&&nz>=0&&nz<GRID_SIZE&&ny>=0&&ny<GRID_LAYERS){
            if(gen.grid[nx][nz][ny]!=CELL_EMPTY) occupied++;
        }
    }
    return 1.0f-(float)occupied/6.0f;
}

static void prepare_instance_data(){
    std::vector<float> offsets,colors,metallics,roughnesses,emissions,alphas,aos;
    std::mt19937 mat_rng(42);
    std::uniform_real_distribution<float> d01(0,1);
    vec3 neon_colors[3]={{0.1f,0.9f,0.9f},{0.9f,0.1f,0.9f},{0.9f,0.9f,0.1f}};
    for(int x=0;x<GRID_SIZE;x++) for(int z=0;z<GRID_SIZE;z++) for(int y=0;y<GRID_LAYERS;y++){
        CellType ct=(CellType)g.gen.grid[x][z][y];
        if(ct==CELL_EMPTY) continue;
        MaterialType mt=CELL_TO_MATERIAL[ct];
        Material mat=MATERIALS[mt];
        if(ct==CELL_FACADE&&d01(mat_rng)<0.15f){
            mat=MATERIALS[MAT_NEON];
            mat.base_color=neon_colors[(int)(d01(mat_rng)*3)%3];
        }
        float nv=simplex::noise3(x*0.1f,y*0.1f,z*0.1f);
        float cv=0.1f*nv;
        vec3 col={clampf(mat.base_color.x+cv,0,1),clampf(mat.base_color.y+cv,0,1),clampf(mat.base_color.z+cv,0,1)};
        float patina=simplex::noise3(x*0.15f,y*0.15f,z*0.15f)*0.15f;
        col=col*(1.0f-fabsf(patina));
        BiomeStratum stratum=get_stratum(y);
        if(stratum==BIOME_UNDERGROUND) col+=vec3(0,0.03f,0);
        float rv=0.1f*simplex::noise3(x*0.15f,y*0.15f,z*0.15f);
        float rough=clampf(mat.roughness+rv,0,1);
        float ao_val=compute_ao(x,z,y,g.gen);
        offsets.push_back((float)x);offsets.push_back((float)y);offsets.push_back((float)z);
        colors.push_back(col.x);colors.push_back(col.y);colors.push_back(col.z);
        metallics.push_back(mat.metallic);
        roughnesses.push_back(rough);
        emissions.push_back(mat.emission);
        alphas.push_back(mat.alpha);
        aos.push_back(ao_val);
    }
    g.instance_count=(int)(offsets.size()/3);
    if(g.instance_count==0) return;
    auto upload=[](GLuint& vbo,const std::vector<float>& data,int loc,int sz){
        glGenBuffers(1,&vbo);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(data.size()*4),data.data(),GL_STATIC_DRAW);
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc,sz,GL_FLOAT,GL_FALSE,0,nullptr);
        glVertexAttribDivisor(loc,1);
    };
    glBindVertexArray(g.cube_vao);
    upload(g.inst_offset_vbo,offsets,2,3);
    upload(g.inst_color_vbo,colors,3,3);
    upload(g.inst_metallic_vbo,metallics,4,1);
    upload(g.inst_roughness_vbo,roughnesses,5,1);
    upload(g.inst_emission_vbo,emissions,6,1);
    upload(g.inst_alpha_vbo,alphas,7,1);
    upload(g.inst_ao_vbo,aos,8,1);
    glBindVertexArray(0);
    printf("Prepared %d cube instances\n",g.instance_count);
}

static void create_cube(){
    std::vector<float> verts;
    std::vector<uint32_t> indices;
    gen_cube_geometry(verts,indices);
    g.cube_index_count=(int)indices.size();
    glGenVertexArrays(1,&g.cube_vao);
    glGenBuffers(1,&g.cube_vbo);
    glGenBuffers(1,&g.cube_ibo);
    glBindVertexArray(g.cube_vao);
    glBindBuffer(GL_ARRAY_BUFFER,g.cube_vbo);
    glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(verts.size()*4),verts.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,24,(void*)0);
    glEnableVertexAttribArray(1);glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,24,(void*)12);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,g.cube_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,(GLsizeiptr)(indices.size()*4),indices.data(),GL_STATIC_DRAW);
    glBindVertexArray(0);
}

static void init_renderer(){
    g.scene_prog=make_program(SCENE_VERT,SCENE_FRAG);
    g.bloom_extract_prog=make_program(QUAD_VERT,BLOOM_EXTRACT_FRAG);
    g.blur_prog=make_program(QUAD_VERT,BLUR_FRAG);
    g.composite_prog=make_program(QUAD_VERT,COMPOSITE_FRAG);
    g.ui_prog=make_program(QUAD_VERT,UI_FRAG);
    create_cube();
    prepare_instance_data();
    create_fbos();
    create_quad();
    create_font_texture();
    g.fog_density=0;g.fog_height_falloff=0.05f;
    g.bloom_threshold=1.5f;g.bloom_intensity=0.2f;g.blur_iterations=2;
    g.enable_postprocessing=false;
    g.inspection_mode=false;g.show_legend=true;
    g.has_selection=false;g.time_val=0;
}

// ray cast
struct RayHit { int x,y,z; bool hit; };
static RayHit ray_cast_mouse(double mx,double my){
    float xndc=(float)(2.0*mx/g.win_w-1.0);
    float yndc=(float)(1.0-2.0*my/g.win_h);
    mat4 proj=mat4_perspective(radiansf(45.f),(float)g.win_w/(float)g.win_h,0.1f,1000.f);
    mat4 view=g.fps_mode?g.fps.get_view():g.orbital.get_view();
    mat4 inv_proj=mat4_inverse(proj);
    mat4 inv_view=mat4_inverse(view);
    vec4 ray_clip(xndc,yndc,-1,1);
    vec4 ray_eye=inv_proj*ray_clip;
    ray_eye=vec4(ray_eye.x,ray_eye.y,-1,0);
    vec4 ray_world=inv_view*ray_eye;
    vec3 ray_dir=normalize(vec3(ray_world.x,ray_world.y,ray_world.z));
    vec3 origin=g.fps_mode?(g.fps.position+vec3(0,FPSCamera::EYE_HEIGHT,0)):g.orbital.position;
    vec3 pos=origin;
    for(int i=0;i<200;i++){
        pos=pos+ray_dir*0.5f;
        int gx=(int)roundf(pos.x),gy=(int)roundf(pos.y),gz=(int)roundf(pos.z);
        if(gx>=0&&gx<GRID_SIZE&&gz>=0&&gz<GRID_SIZE&&gy>=0&&gy<GRID_LAYERS){
            if(g.gen.grid[gx][gz][gy]!=CELL_EMPTY) return{gx,gy,gz,true};
        }
    }
    return{0,0,0,false};
}

static void take_screenshot(){
    mkdir("screenshots",0755);
    time_t now=time(nullptr);
    struct tm* t=localtime(&now);
    char fname[256];
    snprintf(fname,sizeof(fname),"screenshots/gibson_%s_%04d%02d%02d_%02d%02d%02d.tga",
        g.seed.c_str(),t->tm_year+1900,t->tm_mon+1,t->tm_mday,t->tm_hour,t->tm_min,t->tm_sec);
    std::vector<uint8_t> pixels(g.win_w*g.win_h*3);
    glReadPixels(0,0,g.win_w,g.win_h,GL_RGB,GL_UNSIGNED_BYTE,pixels.data());
    if(write_tga(fname,g.win_w,g.win_h,pixels.data())) printf("Screenshot: %s\n",fname);
}

static void render_text_overlay(){
    int tw=g.win_w,th=g.win_h;
    std::vector<uint8_t> overlay(tw*th*4,0);
    auto put_char=[&](int px,int py,char c,uint8_t r,uint8_t gr,uint8_t b){
        int ci=c-32;
        if(ci<0||ci>=96) return;
        for(int row=0;row<8;row++) for(int col=0;col<8;col++){
            if(!((FONT8X8[ci][row]>>(7-col))&1)) continue;
            int sx=px+col*2,sy=py+row*2;
            for(int dy=0;dy<2;dy++) for(int dx=0;dx<2;dx++){
                int fx=sx+dx,fy=sy+dy;
                if(fx<0||fx>=tw||fy<0||fy>=th) continue;
                int idx=((th-1-fy)*tw+fx)*4;
                overlay[idx]=r;overlay[idx+1]=gr;overlay[idx+2]=b;overlay[idx+3]=220;
            }
        }
    };
    auto put_str=[&](int x,int y,const char* s,uint8_t r,uint8_t gr,uint8_t b){
        while(*s){put_char(x,y,*s,r,gr,b);x+=16;s++;}
    };
    auto fill_rect=[&](int x,int y,int w,int h,uint8_t r,uint8_t gr,uint8_t b,uint8_t a){
        for(int py=y;py<y+h&&py<th;py++) for(int px=x;px<x+w&&px<tw;px++){
            int idx=((th-1-py)*tw+px)*4;
            overlay[idx]=r;overlay[idx+1]=gr;overlay[idx+2]=b;overlay[idx+3]=a;
        }
    };
    char buf[128];
    fill_rect(5,5,300,22,0,0,0,160);
    snprintf(buf,sizeof(buf),"Seed: %s",g.seed.c_str());
    put_str(10,8,buf,255,255,255);
    const char* controls[]={
        "Drag:Rotate Wheel:Zoom WASD:Pan",
        "1-5:Presets TAB:FPS R:Regen",
        "S:Screenshot P:PostFX I:Inspect",
        "[]:Fog -=:Bloom L:Legend Q:Quit",
    };
    int cy=30;
    for(auto& line:controls){
        fill_rect(5,cy,500,18,0,0,0,140);
        put_str(10,cy+2,line,200,200,200);
        cy+=20;
    }
    fill_rect(5,cy,200,18,0,0,0,160);
    put_str(10,cy+2,g.fps_mode?"Mode: FPS":"Mode: Orbital",255,255,128);
    cy+=25;
    if(g.enable_postprocessing){
        fill_rect(5,cy,200,18,0,0,0,140);
        snprintf(buf,sizeof(buf),"Fog:%.1f Bloom:%.1f",g.fog_density,g.bloom_intensity);
        put_str(10,cy+2,buf,200,200,200);
        cy+=20;
    }
    if(g.inspection_mode&&g.has_selection){
        fill_rect(5,cy,300,80,0,0,0,180);
        snprintf(buf,sizeof(buf),"INSPECT (%d,%d,%d)",g.sel_x,g.sel_y,g.sel_z);
        put_str(10,cy+2,buf,255,255,128);
        CellType ct=(CellType)g.gen.grid[g.sel_x][g.sel_z][g.sel_y];
        snprintf(buf,sizeof(buf),"Type: %s",CELL_NAMES[ct]);
        put_str(10,cy+20,buf,200,200,200);
        DistrictType dt=g.gen.get_district(g.sel_x,g.sel_z);
        snprintf(buf,sizeof(buf),"Zone: %s",DISTRICT_NAMES[dt]);
        put_str(10,cy+38,buf,200,200,200);
        snprintf(buf,sizeof(buf),"Stratum: %d",get_stratum(g.sel_y));
        put_str(10,cy+56,buf,200,200,200);
    }
    if(g.show_legend){
        int ly=th-180;
        fill_rect(5,ly,180,170,0,0,0,160);
        put_str(10,ly+2,"Materials:",255,255,255);
        const char* mnames[]={"Concrete","Glass","Metal","Neon","Rust","Steel"};
        for(int i=0;i<6;i++){
            vec3 c=MATERIALS[i].base_color;
            int bx=10,by=ly+22+i*22;
            fill_rect(bx,by,16,16,(uint8_t)(c.x*255),(uint8_t)(c.y*255),(uint8_t)(c.z*255),255);
            put_str(bx+22,by+2,mnames[i],200,200,200);
        }
    }
    static GLuint overlay_tex=0;
    if(!overlay_tex){
        glGenTextures(1,&overlay_tex);
        glBindTexture(GL_TEXTURE_2D,overlay_tex);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,tw,th,0,GL_RGBA,GL_UNSIGNED_BYTE,overlay.data());
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    } else {
        glBindTexture(GL_TEXTURE_2D,overlay_tex);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,tw,th,GL_RGBA,GL_UNSIGNED_BYTE,overlay.data());
    }
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(g.ui_prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,overlay_tex);
    glUniform1i(glGetUniformLocation(g.ui_prog,"ui_texture"),0);
    glBindVertexArray(g.quad_vao);
    glDrawArrays(GL_TRIANGLES,0,6);
    glEnable(GL_DEPTH_TEST);
}

static void render_frame(){
    float dt=1.f/60.f;
    g.time_val=(float)glfwGetTime();
    if(!g.fps_mode) g.orbital.update(dt);
    mat4 view=g.fps_mode?g.fps.get_view():g.orbital.get_view();
    mat4 proj=mat4_perspective(radiansf(45.f),(float)g.win_w/(float)g.win_h,0.1f,1000.f);

    // pass 1: scene
    glBindFramebuffer(GL_FRAMEBUFFER,g.scene_fbo);
    glViewport(0,0,g.win_w,g.win_h);
    glClearColor(0.05f,0.05f,0.08f,1);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);glCullFace(GL_BACK);glFrontFace(GL_CCW);
    glEnable(GL_BLEND);glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    if(g.instance_count>0){
        glUseProgram(g.scene_prog);
        glUniformMatrix4fv(glGetUniformLocation(g.scene_prog,"view"),1,GL_FALSE,view.ptr());
        glUniformMatrix4fv(glGetUniformLocation(g.scene_prog,"projection"),1,GL_FALSE,proj.ptr());
        glBindVertexArray(g.cube_vao);
        glDrawElementsInstanced(GL_TRIANGLES,g.cube_index_count,GL_UNSIGNED_INT,nullptr,g.instance_count);
    }

    // pass 2+3: bloom
    if(g.enable_postprocessing){
        glBindFramebuffer(GL_FRAMEBUFFER,g.bloom_extract_fbo);
        glViewport(0,0,g.win_w,g.win_h);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(g.bloom_extract_prog);
        glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,g.scene_color_tex);
        glUniform1i(glGetUniformLocation(g.bloom_extract_prog,"scene_texture"),0);
        glUniform1f(glGetUniformLocation(g.bloom_extract_prog,"bloom_threshold"),g.bloom_threshold);
        glBindVertexArray(g.quad_vao);glDrawArrays(GL_TRIANGLES,0,6);
        for(int i=0;i<g.blur_iterations;i++){
            glBindFramebuffer(GL_FRAMEBUFFER,g.blur_fbo[0]);
            glClear(GL_COLOR_BUFFER_BIT);
            glUseProgram(g.blur_prog);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D,i==0?g.bloom_extract_tex:g.blur_tex[1]);
            glUniform1i(glGetUniformLocation(g.blur_prog,"input_texture"),0);
            glUniform2f(glGetUniformLocation(g.blur_prog,"blur_direction"),1,0);
            glUniform2f(glGetUniformLocation(g.blur_prog,"texture_size"),(float)g.win_w,(float)g.win_h);
            glBindVertexArray(g.quad_vao);glDrawArrays(GL_TRIANGLES,0,6);
            glBindFramebuffer(GL_FRAMEBUFFER,g.blur_fbo[1]);
            glClear(GL_COLOR_BUFFER_BIT);
            glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,g.blur_tex[0]);
            glUniform1i(glGetUniformLocation(g.blur_prog,"input_texture"),0);
            glUniform2f(glGetUniformLocation(g.blur_prog,"blur_direction"),0,1);
            glBindVertexArray(g.quad_vao);glDrawArrays(GL_TRIANGLES,0,6);
        }
    }

    // pass 4: composite
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glViewport(0,0,g.win_w,g.win_h);
    glClearColor(0.05f,0.05f,0.08f,1);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glUseProgram(g.composite_prog);
    glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,g.scene_color_tex);
    glUniform1i(glGetUniformLocation(g.composite_prog,"scene_texture"),0);
    glActiveTexture(GL_TEXTURE1);glBindTexture(GL_TEXTURE_2D,g.scene_depth_tex);
    glUniform1i(glGetUniformLocation(g.composite_prog,"depth_texture"),1);
    glActiveTexture(GL_TEXTURE2);glBindTexture(GL_TEXTURE_2D,g.scene_worldpos_tex);
    glUniform1i(glGetUniformLocation(g.composite_prog,"world_pos_texture"),2);
    glActiveTexture(GL_TEXTURE3);glBindTexture(GL_TEXTURE_2D,g.blur_tex[1]);
    glUniform1i(glGetUniformLocation(g.composite_prog,"bloom_texture"),3);
    glUniform1f(glGetUniformLocation(g.composite_prog,"fog_density"),g.fog_density);
    glUniform3f(glGetUniformLocation(g.composite_prog,"fog_color"),0.1f,0.1f,0.12f);
    glUniform1f(glGetUniformLocation(g.composite_prog,"fog_height_falloff"),g.fog_height_falloff);
    glUniform1f(glGetUniformLocation(g.composite_prog,"bloom_intensity"),g.bloom_intensity);
    glUniform1i(glGetUniformLocation(g.composite_prog,"enable_postprocessing"),g.enable_postprocessing?1:0);
    glUniform1f(glGetUniformLocation(g.composite_prog,"time"),g.time_val);
    glBindVertexArray(g.quad_vao);glDrawArrays(GL_TRIANGLES,0,6);

    render_text_overlay();
}

// ============================================================
// [15] INPUT + MAIN LOOP
// ============================================================
static void regenerate(){
    std::mt19937 srng((uint32_t)time(nullptr));
    std::uniform_int_distribution<int> cd(0,35);
    const char* chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    g.seed.clear();
    for(int i=0;i<8;i++) g.seed+=chars[cd(srng)];
    printf("Regenerating with seed: %s\n",g.seed.c_str());
    g.gen.init(g.seed);
    g.gen.generate();
    glDeleteBuffers(1,&g.inst_offset_vbo);
    glDeleteBuffers(1,&g.inst_color_vbo);
    glDeleteBuffers(1,&g.inst_metallic_vbo);
    glDeleteBuffers(1,&g.inst_roughness_vbo);
    glDeleteBuffers(1,&g.inst_emission_vbo);
    glDeleteBuffers(1,&g.inst_alpha_vbo);
    glDeleteBuffers(1,&g.inst_ao_vbo);
    prepare_instance_data();
    float cx=GRID_SIZE/2.f,cy=GRID_LAYERS/2.f,cz=GRID_SIZE/2.f;
    float dist=std::max((float)GRID_SIZE,(float)GRID_LAYERS)*1.5f;
    g.orbital.init(vec3(cx,cy,cz),dist);
    g.fps.init(vec3(cx,GRID_LAYERS/2.f+2,cz));
}

static void key_callback(GLFWwindow* w,int key,int /*scancode*/,int action,int /*mods*/){
    if(key<0||key>=512) return;
    if(action==GLFW_PRESS){
        g.keys[key]=true;
        if(key==GLFW_KEY_ESCAPE||key==GLFW_KEY_Q) glfwSetWindowShouldClose(w,GLFW_TRUE);
        if(key==GLFW_KEY_TAB){
            g.fps_mode=!g.fps_mode;
            if(g.fps_mode){
                glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_DISABLED);
                g.fps.init(vec3(GRID_SIZE/2.f,(float)GRID_LAYERS/2.f+2,GRID_SIZE/2.f));
            } else glfwSetInputMode(w,GLFW_CURSOR,GLFW_CURSOR_NORMAL);
        }
        if(key==GLFW_KEY_R) regenerate();
        if(key==GLFW_KEY_S) take_screenshot();
        if(key==GLFW_KEY_P) g.enable_postprocessing=!g.enable_postprocessing;
        if(key==GLFW_KEY_I){g.inspection_mode=!g.inspection_mode;if(!g.inspection_mode)g.has_selection=false;}
        if(key==GLFW_KEY_L) g.show_legend=!g.show_legend;
        if(key>=GLFW_KEY_1&&key<=GLFW_KEY_5) g.orbital.set_preset(key-GLFW_KEY_1);
        if(key==GLFW_KEY_SPACE&&g.fps_mode) g.fps.jump();
    }
    if(action==GLFW_RELEASE) g.keys[key]=false;
}

static void mouse_button_callback(GLFWwindow* w,int button,int action,int /*mods*/){
    if(button==GLFW_MOUSE_BUTTON_LEFT){
        if(action==GLFW_PRESS){
            if(g.inspection_mode&&!g.fps_mode){
                double mx,my;glfwGetCursorPos(w,&mx,&my);
                auto hit=ray_cast_mouse(mx,my);
                if(hit.hit){g.sel_x=hit.x;g.sel_y=hit.y;g.sel_z=hit.z;g.has_selection=true;}
            } else if(!g.fps_mode){
                g.mouse_dragging=true;
                glfwGetCursorPos(w,&g.last_mx,&g.last_my);
            }
        }
        if(action==GLFW_RELEASE) g.mouse_dragging=false;
    }
}

static void cursor_pos_callback(GLFWwindow* /*w*/,double xpos,double ypos){
    if(g.fps_mode){
        static bool first=true;
        static double lx=0,ly=0;
        if(first){lx=xpos;ly=ypos;first=false;return;}
        g.fps.mouse_move((float)(xpos-lx),(float)(ypos-ly));
        lx=xpos;ly=ypos;
    } else if(g.mouse_dragging){
        double dx=xpos-g.last_mx,dy=ypos-g.last_my;
        g.orbital.rotate((float)(-dx*0.3),(float)(-dy*0.3));
        g.last_mx=xpos;g.last_my=ypos;
    }
}

static void scroll_callback(GLFWwindow* /*w*/,double /*xoff*/,double yoff){
    if(!g.fps_mode) g.orbital.zoom((float)(-yoff*3));
}

// ============================================================
// [16] SEED UTILITIES + MAIN
// ============================================================
static std::string generate_seed(){
    std::mt19937 srng((uint32_t)time(nullptr));
    std::uniform_int_distribution<int> cd(0,35);
    const char* chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string s;
    for(int i=0;i<8;i++) s+=chars[cd(srng)];
    return s;
}

static bool validate_seed(const std::string& s){
    if(s.size()!=8) return false;
    for(char c:s) if(!((c>='A'&&c<='Z')||(c>='0'&&c<='9'))) return false;
    return true;
}

int main(int argc,char** argv){
    if(argc>1){
        g.seed=argv[1];
        for(auto& c:g.seed) c=(char)toupper(c);
        if(!validate_seed(g.seed)){
            fprintf(stderr,"Invalid seed '%s'. Must be 8 alphanumeric chars.\n",argv[1]);
            return 1;
        }
    } else g.seed=generate_seed();
    printf("Gibson C++: generating with seed %s\n",g.seed.c_str());
    g.gen.init(g.seed);
    g.gen.generate();
    if(!glfwInit()){fprintf(stderr,"GLFW init failed\n");return 1;}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE);
    GLFWmonitor* mon=glfwGetPrimaryMonitor();
    const GLFWvidmode* mode=glfwGetVideoMode(mon);
    g.win_w=mode->width;g.win_h=mode->height;
    g.window=glfwCreateWindow(g.win_w,g.win_h,"Gibson C++",mon,nullptr);
    if(!g.window){fprintf(stderr,"Window creation failed\n");glfwTerminate();return 1;}
    glfwMakeContextCurrent(g.window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(g.window,key_callback);
    glfwSetMouseButtonCallback(g.window,mouse_button_callback);
    glfwSetCursorPosCallback(g.window,cursor_pos_callback);
    glfwSetScrollCallback(g.window,scroll_callback);
    init_renderer();
    float cx=GRID_SIZE/2.f,cy=GRID_LAYERS/2.f,cz=GRID_SIZE/2.f;
    float dist=std::max((float)GRID_SIZE,(float)GRID_LAYERS)*1.5f;
    g.orbital.init(vec3(cx,cy,cz),dist);
    g.fps.init(vec3(cx,cy+2,cz));
    g.fps_mode=false;
    memset(g.keys,0,sizeof(g.keys));
    g.mouse_dragging=false;
    printf("Gibson C++: rendering %d instances\n",g.instance_count);
    while(!glfwWindowShouldClose(g.window)){
        glfwPollEvents();
        if(!g.fps_mode){
            if(g.keys[GLFW_KEY_W]) g.orbital.pan(0,1);
            if(g.keys[GLFW_KEY_S]) g.orbital.pan(0,-1);
            if(g.keys[GLFW_KEY_A]) g.orbital.pan(-1,0);
            if(g.keys[GLFW_KEY_D]) g.orbital.pan(1,0);
        } else {
            float mf=0,mr=0;
            if(g.keys[GLFW_KEY_W]) mf=1;
            if(g.keys[GLFW_KEY_S]) mf=-1;
            if(g.keys[GLFW_KEY_A]) mr=-1;
            if(g.keys[GLFW_KEY_D]) mr=1;
            g.fps.update(1.f/60.f,mf,mr,g.gen.grid);
        }
        if(g.keys[GLFW_KEY_LEFT_BRACKET]) g.fog_density=std::max(0.f,g.fog_density-0.01f);
        if(g.keys[GLFW_KEY_RIGHT_BRACKET]) g.fog_density=std::min(2.f,g.fog_density+0.01f);
        if(g.keys[GLFW_KEY_MINUS]) g.bloom_intensity=std::max(0.f,g.bloom_intensity-0.01f);
        if(g.keys[GLFW_KEY_EQUAL]) g.bloom_intensity=std::min(2.f,g.bloom_intensity+0.01f);
        render_frame();
        glfwSwapBuffers(g.window);
    }
    glfwDestroyWindow(g.window);
    glfwTerminate();
    return 0;
}

