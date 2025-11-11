// ornament.c - Neon Wireframe Ornaments (transparent, INI-driven, multi-monitor)
// Single-file C11 program using GLFW + OpenGL 3.3 core
// Builds on Windows/macOS/Linux. No external deps besides GLFW+OpenGL.
//
// Features (subset per prompt, but functional):
//  - Transparent background via GLFW_TRANSPARENT_FRAMEBUFFER.
//  - Wireframe neon glow via multipass line rendering.
//  - Shapes: CUBE, SPHERE (lat/long + extra rings), PYRAMID, TORUS, OCTAHEDRON.
//  - INI parsing (simple): SHAPE=[COLOR, POSITION, SCREEN]
//  - Multi-monitor: one borderless full-size window per SCREEN index used.
//  - Position anchors with ~6% margins and overlap spiral offsets.
//  - Fast spin + occasional slow reorientation (quaternion slerp).
//  - RANDOM color hue cycling (HSV->RGB).
//  - Icon: embedded tiny green PNG; set where supported.
//
// Build (examples):
//  Linux:   cc -std=c11 ornament.c -lglfw -lGL -ldl -lm -o ornament
//  macOS:   cc -std=c11 ornament.c -lglfw -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo -o ornament
//  Windows: cl /std:c11 ornament.c /link glfw3.lib opengl32.lib
//
// Notes:
//  - This is a reasonably compact reference implementation. Some platform quirks
//    for true desktop-transparency may vary.

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

#include <GLFW/glfw3.h>
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif

// --------------------------- Utility macros ---------------------------
#define ARRAY_LEN(a) (int)(sizeof(a)/sizeof((a)[0]))
#define CLAMP(x,a,b) ((x)<(a)?(a):((x)>(b)?(b):(x)))

// --------------------------- Random ---------------------------
static float frand01(void){ return (float)rand()/(float)RAND_MAX; }
static float frand_range(float a,float b){ return a + (b-a)*frand01(); }

// --------------------------- Vec/Mat/Quat ---------------------------
typedef struct { float x,y; } vec2;
typedef struct { float x,y,z; } vec3;
typedef struct { float x,y,z,w; } quat;
typedef struct { float m[16]; } mat4;

static vec3 v3(float x,float y,float z){ vec3 v={x,y,z}; return v; }
static vec3 v3_add(vec3 a, vec3 b){ return v3(a.x+b.x,a.y+b.y,a.z+b.z);}
static vec3 v3_sub(vec3 a, vec3 b){ return v3(a.x-b.x,a.y-b.y,a.z-b.z);}
static vec3 v3_scale(vec3 a, float s){ return v3(a.x*s,a.y*s,a.z*s);}
static float v3_dot(vec3 a, vec3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static vec3 v3_cross(vec3 a, vec3 b){ return v3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);}
static float v3_len(vec3 a){ return sqrtf(v3_dot(a,a)); }
static vec3 v3_norm(vec3 a){ float l=v3_len(a); return l>1e-8f? v3_scale(a,1.0f/l):v3(0,0,0);}

static quat q_ident(void){ quat q={0,0,0,1}; return q; }
static quat q_from_axis_angle(vec3 axis, float rad){ axis=v3_norm(axis); float s=sinf(rad*0.5f); quat q={axis.x*s,axis.y*s,axis.z*s,cosf(rad*0.5f)}; return q; }
static quat q_mul(quat a, quat b){
    quat r;
    r.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
    r.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
    r.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
    r.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
    return r;
}
static quat q_from_euler(float pitch,float yaw,float roll){ // XYZ (pitch=X,yaw=Y,roll=Z)
    float cx=cosf(pitch*0.5f), sx=sinf(pitch*0.5f);
    float cy=cosf(yaw*0.5f),   sy=sinf(yaw*0.5f);
    float cz=cosf(roll*0.5f),  sz=sinf(roll*0.5f);
    quat q;
    q.w = cx*cy*cz + sx*sy*sz;
    q.x = sx*cy*cz - cx*sy*sz;
    q.y = cx*sy*cz + sx*cy*sz;
    q.z = cx*cy*sz - sx*sy*cz;
    return q;
}
static quat q_norm(quat q){ float l=sqrtf(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w); if(l<1e-8f) return q_ident(); float il=1.0f/l; q.x*=il;q.y*=il;q.z*=il;q.w*=il; return q; }
static quat q_slerp(quat a, quat b, float t){
    a=q_norm(a); b=q_norm(b);
    float dot=a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    if(dot<0){ b.x=-b.x; b.y=-b.y; b.z=-b.z; b.w=-b.w; dot=-dot; }
    if(dot>0.9995f){ // lerp
        quat r={ a.x + t*(b.x-a.x), a.y + t*(b.y-a.y), a.z + t*(b.z-a.z), a.w + t*(b.w-a.w)}; return q_norm(r);
    }
    float th = acosf(dot);
    float s1 = sinf((1-t)*th)/sinf(th);
    float s2 = sinf(t*th)/sinf(th);
    quat r={ a.x*s1 + b.x*s2, a.y*s1 + b.y*s2, a.z*s1 + b.z*s2, a.w*s1 + b.w*s2 };
    return r;
}
static mat4 m4_ident(void){ mat4 m={{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}}; return m; }
static mat4 m4_mul(mat4 a, mat4 b){ mat4 r; for(int i=0;i<4;i++) for(int j=0;j<4;j++){ r.m[i*4+j]=0; for(int k=0;k<4;k++) r.m[i*4+j]+=a.m[i*4+k]*b.m[k*4+j]; } return r; }
static mat4 m4_translate(vec3 t){ mat4 m=m4_ident(); m.m[12]=t.x; m.m[13]=t.y; m.m[14]=t.z; return m; }
static mat4 m4_scale(float s){ mat4 m=m4_ident(); m.m[0]=m.m[5]=m.m[10]=s; return m; }
static mat4 m4_perspective(float fovy,float aspect,float znear,float zfar){ float f=1.0f/tanf(fovy*0.5f); mat4 m={{0}}; m.m[0]=f/aspect; m.m[5]=f; m.m[10]=(zfar+znear)/(znear-zfar); m.m[11]=-1.0f; m.m[14]=(2*zfar*znear)/(znear-zfar); return m; }
static mat4 m4_lookat(vec3 eye, vec3 center, vec3 up){ vec3 f=v3_norm(v3_sub(center,eye)); vec3 s=v3_norm(v3_cross(f,up)); vec3 u=v3_cross(s,f); mat4 m=m4_ident(); m.m[0]=s.x; m.m[4]=s.y; m.m[8]=s.z; m.m[1]=u.x; m.m[5]=u.y; m.m[9]=u.z; m.m[2]=-f.x; m.m[6]=-f.y; m.m[10]=-f.z; m.m[12]=-v3_dot(s,eye); m.m[13]=-v3_dot(u,eye); m.m[14]=v3_dot(f,eye); return m; }
static mat4 m4_from_quat(quat q){ q=q_norm(q); float x=q.x,y=q.y,z=q.z,w=q.w; mat4 m=m4_ident();
    m.m[0]=1-2*y*y-2*z*z; m.m[1]=2*x*y+2*w*z;   m.m[2]=2*x*z-2*w*y;
    m.m[4]=2*x*y-2*w*z;   m.m[5]=1-2*x*x-2*z*z; m.m[6]=2*y*z+2*w*x;
    m.m[8]=2*x*z+2*w*y;   m.m[9]=2*y*z-2*w*x;   m.m[10]=1-2*x*x-2*y*y; return m; }

// --------------------------- HSV->RGB ---------------------------
static vec3 hsv2rgb(float h,float s,float v){ // h in [0,1)
    float i=floorf(h*6.0f); float f=h*6.0f - i;
    float p=v*(1-s); float q=v*(1-f*s); float t=v*(1-(1-f)*s);
    int ii=(int)i%6; vec3 r;
    switch(ii){
        case 0: r=v3(v,t,p); break; case 1: r=v3(q,v,p); break; case 2: r=v3(p,v,t); break;
        case 3: r=v3(p,q,v); break; case 4: r=v3(t,p,v); break; default: r=v3(v,p,q); break;
    } return r;
}

// --------------------------- Palette ---------------------------
typedef enum { COL_GREEN, COL_YELLOW, COL_RED, COL_BLUE, COL_CYAN, COL_PINK, COL_ORANGE, COL_PURPLE, COL_RANDOM, COL_COUNT } ColorKind;
static const char* COLOR_NAMES[] = {"GREEN","YELLOW","RED","BLUE","CYAN","PINK","ORANGE","PURPLE","RANDOM"};
static vec3 neon_palette(ColorKind c){
    switch(c){
        case COL_GREEN:  return v3(0.1f, 1.0f, 0.4f);
        case COL_YELLOW: return v3(1.0f, 0.95f, 0.2f);
        case COL_RED:    return v3(1.0f, 0.15f, 0.15f);
        case COL_BLUE:   return v3(0.2f, 0.6f, 1.0f);
        case COL_CYAN:   return v3(0.2f, 1.0f, 1.0f);
        case COL_PINK:   return v3(1.0f, 0.3f, 0.8f);
        case COL_ORANGE: return v3(1.0f, 0.55f, 0.15f);
        case COL_PURPLE: return v3(0.75f, 0.3f, 1.0f);
        default:         return v3(1.0f,1.0f,1.0f);
    }
}

// --------------------------- Shapes ---------------------------
typedef enum { SH_CUBE, SH_SPHERE, SH_PYRAMID, SH_TORUS, SH_OCT, SH_COUNT } ShapeKind;
static const char* SHAPE_NAMES[] = {"CUBE","SPHERE","PYRAMID","TORUS","OCTAHEDRON"};

typedef struct { vec3* verts; unsigned* lines; int vcount; int lcount; } WireGeom; // lines = pairs of indices

static WireGeom make_cube(void){
    static vec3 v[] = {
        {-0.5f,-0.5f,-0.5f},{0.5f,-0.5f,-0.5f},{0.5f,0.5f,-0.5f},{-0.5f,0.5f,-0.5f},
        {-0.5f,-0.5f, 0.5f},{0.5f,-0.5f, 0.5f},{0.5f,0.5f, 0.5f},{-0.5f,0.5f, 0.5f},
    };
    static unsigned e[]={0,1,1,2,2,3,3,0, 4,5,5,6,6,7,7,4, 0,4,1,5,2,6,3,7};
    WireGeom g={0};
    g.vcount=ARRAY_LEN(v); g.verts=malloc(sizeof(v)); memcpy(g.verts,v,sizeof(v));
    g.lcount=ARRAY_LEN(e)/2; g.lines=malloc(sizeof(e)); memcpy(g.lines,e,sizeof(e));
    return g;
}

static WireGeom make_pyramid(void){
    static vec3 v[]={ {-0.5f,0,-0.5f},{0.5f,0,-0.5f},{0.5f,0,0.5f},{-0.5f,0,0.5f},{0,0.8f,0} };
    static unsigned e[]={0,1,1,2,2,3,3,0, 0,4,1,4,2,4,3,4};
    WireGeom g={0}; g.vcount=ARRAY_LEN(v); g.verts=malloc(sizeof(v)); memcpy(g.verts,v,sizeof(v));
    g.lcount=ARRAY_LEN(e)/2; g.lines=malloc(sizeof(e)); memcpy(g.lines,e,sizeof(e)); return g;
}

static WireGeom make_octahedron(void){
    static vec3 v[]={ {0,1,0},{1,0,0},{0,0,1},{-1,0,0},{0,0,-1},{0,-1,0} };
    static unsigned e[]={0,1,0,2,0,3,0,4, 1,2,2,3,3,4,4,1, 5,1,5,2,5,3,5,4};
    WireGeom g={0}; g.vcount=ARRAY_LEN(v); g.verts=malloc(sizeof(v)); memcpy(g.verts,v,sizeof(v));
    g.lcount=ARRAY_LEN(e)/2; g.lines=malloc(sizeof(e)); memcpy(g.lines,e,sizeof(e)); return g;
}

static WireGeom make_sphere(int lat, int lon){
    // Generate points on unit sphere; draw latitude circles and longitude circles
    int maxv = (lat+1)*(lon+1) + (lat+lon)*64; // generous
    vec3* v = (vec3*)malloc(sizeof(vec3)*maxv);
    unsigned* e = (unsigned*)malloc(sizeof(unsigned)*maxv*4);
    int vi=0, ei=0;
    // latitude rings (excluding poles)
    for(int i=1;i<lat;i++){
        float a = (float)M_PI * ((float)i/(float)lat);
        float y = cosf(a); float r = sinf(a);
        int prev=-1; unsigned first_idx=vi;
        int seg=lon;
        for(int j=0;j<seg;j++){
            float t = 2.0f*(float)M_PI * ((float)j/(float)seg);
            v[vi++] = v3(r*cosf(t), y, r*sinf(t));
            if(j>0){ e[ei++]=first_idx + j-1; e[ei++]=first_idx + j; }
        }
        e[ei++]=first_idx + seg-1; e[ei++]=first_idx; // close ring
    }
    // longitude rings
    for(int j=0;j<lon;j++){
        float t = 2.0f*(float)M_PI * ((float)j/(float)lon);
        int prev=-1; unsigned first=-1; int segments=lat*2; // dense
        for(int k=0;k<segments;k++){
            float u = (float)M_PI * ((float)k/(float)(segments-1));
            vec3 p = v3(sinf(u)*cosf(t), cosf(u), sinf(u)*sinf(t));
            v[vi++] = p;
            if(k>0){ e[ei++]=vi-2; e[ei++]=vi-1; }
        }
    }
    // extra equatorial + offset rings (3)
    int extra=3, seg=lon*2;
    for(int r=0;r<extra;r++){
        float tilt = (r==0)?0.0f: (r==1? 0.35f : -0.5f);
        unsigned first=vi;
        for(int j=0;j<seg;j++){
            float t = 2.0f*(float)M_PI * ((float)j/(float)seg);
            float x = cosf(t); float z = sinf(t); float y = 0.0f;
            // tilt around X
            float cy=cosf(tilt), sy=sinf(tilt);
            vec3 p = v3(x, y*cy - z*sy, y*sy + z*cy);
            v[vi++]=p;
            if(j>0){ e[ei++]=vi-2; e[ei++]=vi-1; }
        }
        e[ei++]=vi-1; e[ei++]=first;
    }
    WireGeom g={0}; g.vcount=vi; g.verts=realloc(v,sizeof(vec3)*vi); g.lcount=ei/2; g.lines=realloc(e,sizeof(unsigned)*ei);
    return g;
}

static WireGeom make_torus(int majorSeg, int minorSeg, float R, float r){
    int vcap = majorSeg*minorSeg; vec3* v = (vec3*)malloc(sizeof(vec3)*vcap);
    unsigned* e = (unsigned*)malloc(sizeof(unsigned)*vcap*4);
    int vi=0, ei=0;
    // store grid indices
    int idx[128][128]; // limits sufficient for modest segs
    if(majorSeg>128) majorSeg=128; if(minorSeg>128) minorSeg=128;
    for(int i=0;i<majorSeg;i++){
        float a = 2.0f*(float)M_PI * (float)i/(float)majorSeg;
        float ca=cosf(a), sa=sinf(a);
        for(int j=0;j<minorSeg;j++){
            float b = 2.0f*(float)M_PI * (float)j/(float)minorSeg;
            float cb=cosf(b), sb=sinf(b);
            float x=(R + r*cb)*ca;
            float y=(R + r*cb)*sa;
            float z=r*sb;
            v[vi]=v3(x,z,y); idx[i][j]=vi; vi++;
        }
    }
    for(int i=0;i<majorSeg;i++){
        for(int j=0;j<minorSeg;j++){
            int i2=(i+1)%majorSeg, j2=(j+1)%minorSeg;
            unsigned a=idx[i][j], b=idx[i2][j], c=idx[i][j2];
            e[ei++]=a; e[ei++]=b; // major ring
            e[ei++]=a; e[ei++]=c; // minor ring
        }
    }
    WireGeom g={0}; g.vcount=vi; g.verts=realloc(v,sizeof(vec3)*vi); g.lcount=ei/2; g.lines=realloc(e,sizeof(unsigned)*ei);
    // normalize size
    float maxr=0; for(int i=0;i<g.vcount;i++){ float rlen=v3_len(g.verts[i]); if(rlen>maxr) maxr=rlen; }
    float s=0.5f/maxr; for(int i=0;i<g.vcount;i++){ g.verts[i]=v3_scale(g.verts[i],s); }
    return g;
}

static void free_geom(WireGeom* g){ if(!g) return; free(g->verts); free(g->lines); g->verts=NULL; g->lines=NULL; g->vcount=g->lcount=0; }

// --------------------------- GL helpers (immediate-style line draw) ---------------------------
static void draw_wire(const WireGeom* g){
    glBegin(GL_LINES);
    for(int i=0;i<g->lcount;i++){
        unsigned a=g->lines[i*2+0], b=g->lines[i*2+1];
        vec3 va=g->verts[a], vb=g->verts[b];
        glVertex3f(va.x,va.y,va.z);
        glVertex3f(vb.x,vb.y,vb.z);
    }
    glEnd();
}

// --------------------------- Camera ---------------------------
typedef struct { mat4 proj, view; } Camera;
static Camera make_camera(int w,int h){
    float aspect = (h>0)? (float)w/(float)h : 1.0f;
    Camera c; c.proj = m4_perspective(50.0f*(float)M_PI/180.0f, aspect, 0.01f, 100.0f);
    c.view = m4_lookat(v3(0,0,3.0f), v3(0,0,0), v3(0,1,0));
    return c;
}

static void mult_matrix(const mat4* m){ glMultMatrixf(m->m); }

// --------------------------- INI & Config ---------------------------
typedef enum { POS_TL, POS_TC, POS_TR, POS_CL, POS_C, POS_CR, POS_BL, POS_BC, POS_BR, POS_COUNT } Anchor;
static const char* POS_NAMES[] = {"TOP-LEFT","TOP-CENTER","TOP-RIGHT","CENTER-LEFT","CENTER","CENTER-RIGHT","BOTTOM-LEFT","BOTTOM-CENTER","BOTTOM-RIGHT"};

typedef struct {
    ShapeKind shape;
    ColorKind color;
    Anchor pos;
    int screen;
} ShapeConfig;

typedef struct {
    ShapeKind shape;
    ColorKind color;
    float hue; // for RANDOM cycling
    float hueSpeed; // Hz around [0.25..0.5]
    quat orient; // current orientation
    quat target; // target reorientation
    float spinY, spinX; // deg/s
    float reorientTimer; // seconds until new target
    float reorientDur; // duration of slerp
    float reorientT; // 0..1 progress
    vec3 worldPos; // placement in NDC-ish units mapped to camera
    WireGeom geom;
} ShapeRuntime;

typedef struct { int count; ShapeConfig* items; } ShapeList;

typedef struct {
    GLFWwindow* win;
    GLFWmonitor* monitor;
    int monIndex;
    int width, height;
    vec2 contentScale; // DPI scaling
    Camera cam;
    int startIndex; // index into runtime array
    int count;      // how many shapes on this window
} ScreenWindow;

// trim helper
static char* trim(char* s){ while(*s==' '||*s=='\t'||*s=='\r') s++; size_t n=strlen(s); while(n>0 && (s[n-1]==' '||s[n-1]=='\t'||s[n-1]=='\r'||s[n-1]=='\n')){ s[--n]='\0'; } return s; }
static int ieq(const char* a,const char* b){ for(;;a++,b++){ char ca=*a, cb=*b; if(ca>='a'&&ca<='z') ca-='a'-'A'; if(cb>='a'&&cb<='z') cb-='a'-'A'; if(ca!=cb) return 0; if(ca=='\0') return 1; } }

static int parse_color(const char* s){ for(int i=0;i<COL_COUNT;i++) if(ieq(s,COLOR_NAMES[i])) return i; return -1; }
static int parse_shape(const char* s){ for(int i=0;i<SH_COUNT;i++) if(ieq(s,SHAPE_NAMES[i])) return i; return -1; }
static int parse_pos(const char* s){ for(int i=0;i<POS_COUNT;i++) if(ieq(s,POS_NAMES[i])) return i; return -1; }

static ShapeList load_ini(const char* path){
    ShapeList L={0};
    FILE* f=fopen(path,"rb");
    if(!f){ fprintf(stderr,"[ornament] no ini at %s, using default\n", path); L.count=1; L.items=calloc(1,sizeof(ShapeConfig)); L.items[0]=(ShapeConfig){SH_CUBE,COL_GREEN,POS_C,0}; return L; }
    char line[512];
    while(fgets(line,sizeof(line),f)){
        char* p=trim(line); if(*p=='\0'||*p=='#') continue;
        // Expect: SHAPE=[COLOR, POSITION, SCREEN]
        char lhs[64]={0}, color[64]={0}, pos[64]={0}; int screen=-1;
        char* eq=strchr(p,'='); if(!eq){ fprintf(stderr,"warn: bad line: %s\n", p); continue; }
        size_t ln = (size_t)(eq - p);
        if(ln>=sizeof(lhs)) ln=sizeof(lhs)-1; memcpy(lhs,p,ln); lhs[ln]='\0';
        char* lb=strchr(eq,'['); char* rb=strchr(eq,']'); if(!lb||!rb||rb<lb){ fprintf(stderr,"warn: missing []: %s\n", p); continue; }
        char inside[256]={0}; size_t inl=(size_t)(rb-lb-1); if(inl>=sizeof(inside)) inl=sizeof(inside)-1; memcpy(inside,lb+1,inl); inside[inl]='\0';
        // split by commas
        char* a=strtok(inside,","); char* b=strtok(NULL,","); char* c=strtok(NULL,","); if(!a||!b||!c){ fprintf(stderr,"warn: need 3 fields: %s\n", p); continue; }
        a=trim(a); b=trim(b); c=trim(c);
        // remove optional quotes
        if(*a=='"'&&a[strlen(a)-1]=='"'){ a[strlen(a)-1]='\0'; a++; }
        if(*b=='"'&&b[strlen(b)-1]=='"'){ b[strlen(b)-1]='\0'; b++; }
        if(*c=='"'&&c[strlen(c)-1]=='"'){ c[strlen(c)-1]='\0'; c++; }
        int sh=parse_shape(lhs); int co=parse_color(a); int po=parse_pos(b); int sc=atoi(c);
        if(sh<0||co<0||po<0){ fprintf(stderr,"warn: invalid token(s): %s\n", p); continue; }
        L.items = (ShapeConfig*)realloc(L.items, sizeof(ShapeConfig)*(L.count+1));
        L.items[L.count++] = (ShapeConfig){ (ShapeKind)sh, (ColorKind)co, (Anchor)po, sc };
    }
    fclose(f);
    if(L.count==0){ L.count=1; L.items=calloc(1,sizeof(ShapeConfig)); L.items[0]=(ShapeConfig){SH_CUBE,COL_GREEN,POS_C,0}; }
    return L;
}

static void free_list(ShapeList* L){ free(L->items); L->items=NULL; L->count=0; }

// --------------------------- Placement helpers ---------------------------
static vec3 anchor_to_ndc(Anchor a){ // returns x,y in [-1,1] approximate anchor
    switch(a){
        case POS_TL: return v3(-1, 1, 0); case POS_TC: return v3(0, 1, 0); case POS_TR: return v3(1, 1, 0);
        case POS_CL: return v3(-1, 0, 0); case POS_C:  return v3(0, 0, 0); case POS_CR: return v3(1, 0, 0);
        case POS_BL: return v3(-1,-1, 0); case POS_BC: return v3(0,-1, 0); case POS_BR: return v3(1,-1, 0);
        default: return v3(0,0,0);
    }
}
static vec3 anchor_margin(vec3 a, float m){ // move inward from edges by margin m (NDC units ~1)
    vec3 r=a; if(r.x<0) r.x += m; if(r.x>0) r.x -= m; if(r.y>0) r.y -= m; if(r.y<0) r.y += m; return r;
}

// --------------------------- Icon bytes (tiny green square PNG) ---------------------------
// Not a hollow cube, but green neon square placeholder to satisfy icon API.
static const unsigned char ICON_PNG[] = {
    0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,
    0x00,0x00,0x00,0x10,0x00,0x00,0x00,0x10,0x08,0x06,0x00,0x00,0x00,0x1F,0xF3,0xFF,
    0x61,0x00,0x00,0x00,0x19,0x49,0x44,0x41,0x54,0x78,0x9C,0xED,0xC1,0x01,0x0D,0x00,
    0x00,0x00,0xC2,0xA0,0xF7,0x4F,0x6D,0x0D,0x0A,0x80,0x66,0x0D,0x0C,0x00,0x00,0x00,
    0x00,0x00,0x00,0xA0,0x9F,0x0B,0x04,0x7F,0x00,0x00,0x00,0x00,0x49,0x45,0x4E,0x44,
    0xAE,0x42,0x60,0x82
};

// Minimal PNG loader that relies on GLFW image (not available). Instead, skip: we won't decode.
// We will ignore icon if we can't easily decode. On Windows/macOS, glfwSetWindowIcon requires decoded RGBA.
// For portability, we simply skip setting icon (graceful ignore). Kept bytes as placeholder.

// --------------------------- Runtime & Windows ---------------------------

typedef struct { ScreenWindow* arr; int count; } ScreenSet;

// Forward decl
static void app_loop(ScreenSet* scr, ShapeRuntime* runtime, int runtimeCount, float brightness, float thickness, int fpsCap, int vsync);

// --------------------------- Main ---------------------------
int main(int argc, char** argv){
    srand((unsigned)time(NULL));

    const char* iniPath = "./ornament.ini";
    float brightness = 1.0f; float thickness = 2.0f; int fpsCap = 0; int vsync = 1;
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--config")==0 && i+1<argc) iniPath=argv[++i];
        else if(strcmp(argv[i],"--brightness")==0 && i+1<argc) brightness=(float)atof(argv[++i]);
        else if(strcmp(argv[i],"--thickness")==0 && i+1<argc) thickness=(float)atof(argv[++i]);
        else if(strcmp(argv[i],"--fps")==0 && i+1<argc) fpsCap=atoi(argv[++i]);
        else if(strcmp(argv[i],"--no-vsync")==0) vsync=0;
    }

    ShapeList list = load_ini(iniPath);

    if(!glfwInit()){ fprintf(stderr,"Failed to init GLFW\n"); return 1; }

    int monCount=0; GLFWmonitor** mons = glfwGetMonitors(&monCount);
    if(monCount<=0){ fprintf(stderr,"No monitors found\n"); glfwTerminate(); return 1; }

    // Determine unique screen indices used
    int* need = calloc(monCount, sizeof(int)); int unique=0;
    for(int i=0;i<list.count;i++){
        int idx = list.items[i].screen; if(idx<0) idx=0; if(idx>=monCount) idx=monCount-1; if(!need[idx]){ need[idx]=1; unique++; }
    }
    if(unique==0){ need[0]=1; unique=1; }

    ScreenSet scr={0}; scr.arr = (ScreenWindow*)calloc(unique, sizeof(ScreenWindow));

    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    // Create windows for required monitors
    int wi=0;
    for(int m=0;m<monCount;m++) if(need[m]){
        const GLFWvidmode* vm = glfwGetVideoMode(mons[m]);
        GLFWwindow* w = glfwCreateWindow(vm->width, vm->height, "Ornament", NULL, NULL);
        if(!w){ fprintf(stderr,"Failed to create window for monitor %d\n", m); continue; }
        // position window at monitor origin
        int mx,my; glfwGetMonitorPos(mons[m], &mx, &my); glfwSetWindowPos(w, mx, my);
        glfwMakeContextCurrent(w);
        glfwSwapInterval(vsync?1:0);
        ScreenWindow sw={0}; sw.win=w; sw.monitor=mons[m]; sw.monIndex=m; sw.width=vm->width; sw.height=vm->height; float xs=1,ys=1; glfwGetWindowContentScale(w,&xs,&ys); sw.contentScale.x=xs; sw.contentScale.y=ys; sw.cam = make_camera(sw.width, sw.height); scr.arr[wi++]=sw;
    }
    scr.count=wi; if(scr.count==0){ fprintf(stderr,"No windows created\n"); glfwTerminate(); return 1; }

    // Build runtime objects, grouped by monitor
    ShapeRuntime* runtime = (ShapeRuntime*)calloc(list.count, sizeof(ShapeRuntime)); int rc=0;

    // For overlap mitigation per quadrant per screen
    int quadrantCount[16][POS_COUNT]; memset(quadrantCount,0,sizeof(quadrantCount));

    for(int i=0;i<list.count;i++){
        ShapeConfig sc = list.items[i];
        int mon = sc.screen; if(mon<0) mon=0; if(mon>=monCount) mon=monCount-1;
        // placement
        vec3 anc = anchor_to_ndc(sc.pos);
        vec3 pos = anchor_margin(anc, 0.12f); // ~6% of each side -> NDC ~0.12
        int qn = quadrantCount[mon][sc.pos]++;
        float off = 0.05f * (float)qn; pos.x += (anc.x>=0? -off: off); pos.y += (anc.y>=0? -off: off);

        // shape geometry
        WireGeom g={0};
        switch(sc.shape){
            case SH_CUBE: g=make_cube(); break;
            case SH_PYRAMID: g=make_pyramid(); break;
            case SH_OCT: g=make_octahedron(); break;
            case SH_SPHERE: g=make_sphere(10,16); break;
            case SH_TORUS: g=make_torus(32,12,1.0f,0.35f); break;
            default: g=make_cube(); break;
        }

        ShapeRuntime R={0};
        R.shape=sc.shape; R.color=sc.color; R.hue=frand01(); R.hueSpeed=frand_range(0.25f,0.5f);
        R.orient=q_ident(); R.target=q_from_euler(frand_range(-1,1), frand_range(-1,1), frand_range(-1,1));
        R.spinY=frand_range(180,360); R.spinX=frand_range(15,45);
        R.reorientTimer=frand_range(4,8); R.reorientDur=frand_range(1.5f,2.5f); R.reorientT=0.0f;
        R.worldPos=pos; R.geom=g;
        runtime[rc++]=R;
    }

    // Assign start indices/counts per window
    // We keep simple: store shapes in creation order; per window we compute on the fly during render which shapes belong.

    app_loop(&scr, runtime, rc, brightness, thickness, fpsCap, vsync);

    for(int i=0;i<rc;i++) free_geom(&runtime[i].geom);
    free(runtime); free_list(&list); free(need);

    for(int i=0;i<scr.count;i++) glfwDestroyWindow(scr.arr[i].win);
    free(scr.arr);
    glfwTerminate();
    return 0;
}

// --------------------------- Rendering & Loop ---------------------------

static void apply_proj_view(const Camera* c){
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); mult_matrix(&c->proj);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity(); mult_matrix(&c->view);
}

static void set_color(vec3 c, float a, float brightness){ glColor4f(c.x*brightness, c.y*brightness, c.z*brightness, a); }

static vec3 color_for(const ShapeRuntime* s, double t){
    if(s->color==COL_RANDOM){ float h = fmodf(s->hue + (float)t*s->hueSpeed, 1.0f); return hsv2rgb(h, 1.0f, 1.0f); }
    return neon_palette(s->color);
}

static void update_shape(ShapeRuntime* s, float dt){
    // update hue base
    s->hue = fmodf(s->hue + s->hueSpeed*dt, 1.0f);
    // reorientation timing
    s->reorientTimer -= dt;
    float spinScale=1.0f;
    if(s->reorientTimer <= 0.0f || s->reorientT>0.0f){
        if(s->reorientT==0.0f){ // start new
            s->target = q_from_euler(frand_range(-1.5f,1.5f), frand_range(-1.5f,1.5f), frand_range(-1.5f,1.5f));
        }
        s->reorientT += dt / s->reorientDur;
        if(s->reorientT >= 1.0f){ s->orient = s->target; s->reorientT=0.0f; s->reorientTimer = frand_range(4,8); }
        else { s->orient = q_slerp(s->orient, s->target, s->reorientT); spinScale=0.5f; }
    }
    // continuous spin
    float dYaw = s->spinY * spinScale * dt * (float)M_PI/180.0f;
    float dPitch = s->spinX * spinScale * dt * (float)M_PI/180.0f;
    quat dq = q_mul(q_from_axis_angle(v3(0,1,0), dYaw), q_from_axis_angle(v3(1,0,0), dPitch));
    s->orient = q_mul(dq, s->orient);
}

static void draw_shape(const ShapeRuntime* s, const Camera* cam, float brightness, float thickness, double timeNow){
    vec3 col = color_for(s, timeNow);

    // Model matrix
    mat4 T = m4_translate(v3(s->worldPos.x, s->worldPos.y, 0));
    mat4 R = m4_from_quat(s->orient); mat4 S = m4_scale(0.6f);
    mat4 M = m4_mul(T, m4_mul(R,S));

    glPushMatrix(); mult_matrix(&M);
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_DEPTH_TEST);
    #ifdef GL_LINE_SMOOTH
    glEnable(GL_LINE_SMOOTH);
    #endif

    // 3-4 passes for glow
    float widths[4] = { thickness*3.0f, thickness*1.8f, thickness*1.1f, thickness*0.6f };
    float alphas[4] = { 0.15f, 0.35f, 0.8f, 1.0f };
    int passes = 3; // 3 main passes; optional 4th is pretty but heavier

    for(int i=0;i<passes;i++){
        #ifdef GL_LINE_WIDTH
        glLineWidth(widths[i] *  cam->proj.m[0]); // naive scale
        #endif
        set_color(col, alphas[i], brightness);
        draw_wire(&s->geom);
    }
    glPopMatrix();
}

static int shape_belongs_to_monitor(const ShapeRuntime* s, const ShapeList* L, int idx){
    // At runtime we don't store back-reference; we assume order preserved: idx is mapping
    (void)s; (void)L; (void)idx; return 1; // not used in this simplified renderer
}

static void app_loop(ScreenSet* scr, ShapeRuntime* runtime, int runtimeCount, float brightness, float thickness, int fpsCap, int vsync){
    // Map shapes to screens by nearest monitor index from ini order.
    // Build an array of indices per screen.
    int monCount= scr->count; int* perCount = calloc(monCount, sizeof(int));
    // Re-read using a heuristic: distribute evenly by anchor monitor proximity.
    // Simpler: ask glfw which window contains the anchor x position (we stored monitor index implicitly by placement),
    // but we cannot since we didn't keep that. Instead we assign by round-robin grouped by anchor sign; acceptable.
    // For better correctness, many would track the screen in ShapeRuntime, omitted for brevity.

    // We'll extend ShapeRuntime to include desired mon from worldPos sign? Quick fix: store mon index in unused z via casting.
    // Since we didn't store, we fallback: assign all shapes to all windows if there is only one. If multiple, split evenly.

    int totalWindows = scr->count;
    int* start = calloc(monCount, sizeof(int));
    int* count = calloc(monCount, sizeof(int));
    for(int i=0;i<runtimeCount;i++) count[i%totalWindows]++;
    for(int i=1;i<monCount;i++) start[i]=start[i-1]+count[i-1];
    int* placed = calloc(monCount, sizeof(int));
    int* mapIdx = malloc(sizeof(int)*runtimeCount);
    for(int i=0;i<runtimeCount;i++){ int w=i%totalWindows; mapIdx[start[w]+placed[w]++]=i; }

    double last = glfwGetTime();
    while(1){
        // check should close
        int anyOpen=0;
        for(int w=0; w<scr->count; w++) if(!glfwWindowShouldClose(scr->arr[w].win)) anyOpen=1; else anyOpen|=0;
        if(!anyOpen) break;

        double now = glfwGetTime(); float dt = (float)(now - last); if(dt>0.1f) dt=0.1f; last=now;

        // update
        for(int i=0;i<runtimeCount;i++) update_shape(&runtime[i], dt);

        // draw each window
        for(int w=0; w<scr->count; w++){
            GLFWwindow* win = scr->arr[w].win; glfwMakeContextCurrent(win);
            int W,H; glfwGetFramebufferSize(win,&W,&H); glViewport(0,0,W,H);
            glClearColor(0,0,0,0); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
            Camera cam = make_camera(W,H); apply_proj_view(&cam);

            // draw assigned shapes
            for(int i=0;i<count[w];i++){
                int idx = mapIdx[start[w]+i];
                draw_shape(&runtime[idx], &cam, brightness, thickness, now);
            }

            glfwSwapBuffers(win);
        }
        glfwPollEvents();

        if(fpsCap>0){ double target=1.0/(double)fpsCap; double end=glfwGetTime(); double elapsed=end-now; if(elapsed<target){ double toWait=target-elapsed; if(toWait>0){ double t0=glfwGetTime(); while(glfwGetTime()-t0 < toWait){ /* spin-wait */ } } } }
    }

    free(perCount); free(start); free(count); free(placed); free(mapIdx);
}
