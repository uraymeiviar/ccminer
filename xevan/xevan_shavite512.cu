#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define sph_u32 uint32_t
#define sph_u64 uint64_t

#define AESx(x)   SPH_C32(x)
#define AES0      AES0_S
#define AES1      AES1_S
#define AES2      AES2_S
#define AES3      AES3_S

#define AES_ROUND_LE(X0, X1, X2, X3, K0, K1, K2, K3, Y0, Y1, Y2, Y3)   do { \
  (Y0) = AES0[__byte_perm(X0, 0, 0x4440)] \
    ^ AES1[__byte_perm(X1, 0, 0x4441)] \
    ^ AES2[__byte_perm(X2, 0, 0x4442)] \
    ^ __ldg(&AES3_C[__byte_perm(X3, 0, 0x4443)]) ^ (K0); \
  (Y1) = __ldg(&AES0_C[__byte_perm(X1, 0, 0x4440)]) \
    ^ AES1[__byte_perm(X2, 0, 0x4441)] \
    ^ AES2[__byte_perm(X3, 0, 0x4442)] \
    ^ __ldg(&AES3_C[__byte_perm(X0, 0, 0x4443)]) ^ (K1); \
  (Y2) = AES0[__byte_perm(X2, 0, 0x4440)] \
    ^ AES1[__byte_perm(X3, 0, 0x4441)] \
    ^ AES2[__byte_perm(X0, 0, 0x4442)] \
    ^ __ldg(&AES3_C[__byte_perm(X1, 0, 0x4443)]) ^ (K2); \
  (Y3) = AES0[__byte_perm(X3, 0, 0x4440)] \
    ^ AES1[__byte_perm(X0, 0, 0x4441)] \
    ^ AES2[__byte_perm(X1, 0, 0x4442)] \
    ^ __ldg(&AES3_C[__byte_perm(X2, 0, 0x4443)]) ^ (K3); \
} while (0)

#define AES_ROUND_NOKEY_LE(X0, X1, X2, X3, Y0, Y1, Y2, Y3) \
AES_ROUND_LE(X0, X1, X2, X3, 0, 0, 0, 0, Y0, Y1, Y2, Y3)

__constant__ static const sph_u32 AES0_C[256] = {
    AESx(0xA56363C6), AESx(0x847C7CF8), AESx(0x997777EE), AESx(0x8D7B7BF6),
    AESx(0x0DF2F2FF), AESx(0xBD6B6BD6), AESx(0xB16F6FDE), AESx(0x54C5C591),
    AESx(0x50303060), AESx(0x03010102), AESx(0xA96767CE), AESx(0x7D2B2B56),
    AESx(0x19FEFEE7), AESx(0x62D7D7B5), AESx(0xE6ABAB4D), AESx(0x9A7676EC),
    AESx(0x45CACA8F), AESx(0x9D82821F), AESx(0x40C9C989), AESx(0x877D7DFA),
    AESx(0x15FAFAEF), AESx(0xEB5959B2), AESx(0xC947478E), AESx(0x0BF0F0FB),
    AESx(0xECADAD41), AESx(0x67D4D4B3), AESx(0xFDA2A25F), AESx(0xEAAFAF45),
    AESx(0xBF9C9C23), AESx(0xF7A4A453), AESx(0x967272E4), AESx(0x5BC0C09B),
    AESx(0xC2B7B775), AESx(0x1CFDFDE1), AESx(0xAE93933D), AESx(0x6A26264C),
    AESx(0x5A36366C), AESx(0x413F3F7E), AESx(0x02F7F7F5), AESx(0x4FCCCC83),
    AESx(0x5C343468), AESx(0xF4A5A551), AESx(0x34E5E5D1), AESx(0x08F1F1F9),
    AESx(0x937171E2), AESx(0x73D8D8AB), AESx(0x53313162), AESx(0x3F15152A),
    AESx(0x0C040408), AESx(0x52C7C795), AESx(0x65232346), AESx(0x5EC3C39D),
    AESx(0x28181830), AESx(0xA1969637), AESx(0x0F05050A), AESx(0xB59A9A2F),
    AESx(0x0907070E), AESx(0x36121224), AESx(0x9B80801B), AESx(0x3DE2E2DF),
    AESx(0x26EBEBCD), AESx(0x6927274E), AESx(0xCDB2B27F), AESx(0x9F7575EA),
    AESx(0x1B090912), AESx(0x9E83831D), AESx(0x742C2C58), AESx(0x2E1A1A34),
    AESx(0x2D1B1B36), AESx(0xB26E6EDC), AESx(0xEE5A5AB4), AESx(0xFBA0A05B),
    AESx(0xF65252A4), AESx(0x4D3B3B76), AESx(0x61D6D6B7), AESx(0xCEB3B37D),
    AESx(0x7B292952), AESx(0x3EE3E3DD), AESx(0x712F2F5E), AESx(0x97848413),
    AESx(0xF55353A6), AESx(0x68D1D1B9), AESx(0x00000000), AESx(0x2CEDEDC1),
    AESx(0x60202040), AESx(0x1FFCFCE3), AESx(0xC8B1B179), AESx(0xED5B5BB6),
    AESx(0xBE6A6AD4), AESx(0x46CBCB8D), AESx(0xD9BEBE67), AESx(0x4B393972),
    AESx(0xDE4A4A94), AESx(0xD44C4C98), AESx(0xE85858B0), AESx(0x4ACFCF85),
    AESx(0x6BD0D0BB), AESx(0x2AEFEFC5), AESx(0xE5AAAA4F), AESx(0x16FBFBED),
    AESx(0xC5434386), AESx(0xD74D4D9A), AESx(0x55333366), AESx(0x94858511),
    AESx(0xCF45458A), AESx(0x10F9F9E9), AESx(0x06020204), AESx(0x817F7FFE),
    AESx(0xF05050A0), AESx(0x443C3C78), AESx(0xBA9F9F25), AESx(0xE3A8A84B),
    AESx(0xF35151A2), AESx(0xFEA3A35D), AESx(0xC0404080), AESx(0x8A8F8F05),
    AESx(0xAD92923F), AESx(0xBC9D9D21), AESx(0x48383870), AESx(0x04F5F5F1),
    AESx(0xDFBCBC63), AESx(0xC1B6B677), AESx(0x75DADAAF), AESx(0x63212142),
    AESx(0x30101020), AESx(0x1AFFFFE5), AESx(0x0EF3F3FD), AESx(0x6DD2D2BF),
    AESx(0x4CCDCD81), AESx(0x140C0C18), AESx(0x35131326), AESx(0x2FECECC3),
    AESx(0xE15F5FBE), AESx(0xA2979735), AESx(0xCC444488), AESx(0x3917172E),
    AESx(0x57C4C493), AESx(0xF2A7A755), AESx(0x827E7EFC), AESx(0x473D3D7A),
    AESx(0xAC6464C8), AESx(0xE75D5DBA), AESx(0x2B191932), AESx(0x957373E6),
    AESx(0xA06060C0), AESx(0x98818119), AESx(0xD14F4F9E), AESx(0x7FDCDCA3),
    AESx(0x66222244), AESx(0x7E2A2A54), AESx(0xAB90903B), AESx(0x8388880B),
    AESx(0xCA46468C), AESx(0x29EEEEC7), AESx(0xD3B8B86B), AESx(0x3C141428),
    AESx(0x79DEDEA7), AESx(0xE25E5EBC), AESx(0x1D0B0B16), AESx(0x76DBDBAD),
    AESx(0x3BE0E0DB), AESx(0x56323264), AESx(0x4E3A3A74), AESx(0x1E0A0A14),
    AESx(0xDB494992), AESx(0x0A06060C), AESx(0x6C242448), AESx(0xE45C5CB8),
    AESx(0x5DC2C29F), AESx(0x6ED3D3BD), AESx(0xEFACAC43), AESx(0xA66262C4),
    AESx(0xA8919139), AESx(0xA4959531), AESx(0x37E4E4D3), AESx(0x8B7979F2),
    AESx(0x32E7E7D5), AESx(0x43C8C88B), AESx(0x5937376E), AESx(0xB76D6DDA),
    AESx(0x8C8D8D01), AESx(0x64D5D5B1), AESx(0xD24E4E9C), AESx(0xE0A9A949),
    AESx(0xB46C6CD8), AESx(0xFA5656AC), AESx(0x07F4F4F3), AESx(0x25EAEACF),
    AESx(0xAF6565CA), AESx(0x8E7A7AF4), AESx(0xE9AEAE47), AESx(0x18080810),
    AESx(0xD5BABA6F), AESx(0x887878F0), AESx(0x6F25254A), AESx(0x722E2E5C),
    AESx(0x241C1C38), AESx(0xF1A6A657), AESx(0xC7B4B473), AESx(0x51C6C697),
    AESx(0x23E8E8CB), AESx(0x7CDDDDA1), AESx(0x9C7474E8), AESx(0x211F1F3E),
    AESx(0xDD4B4B96), AESx(0xDCBDBD61), AESx(0x868B8B0D), AESx(0x858A8A0F),
    AESx(0x907070E0), AESx(0x423E3E7C), AESx(0xC4B5B571), AESx(0xAA6666CC),
    AESx(0xD8484890), AESx(0x05030306), AESx(0x01F6F6F7), AESx(0x120E0E1C),
    AESx(0xA36161C2), AESx(0x5F35356A), AESx(0xF95757AE), AESx(0xD0B9B969),
    AESx(0x91868617), AESx(0x58C1C199), AESx(0x271D1D3A), AESx(0xB99E9E27),
    AESx(0x38E1E1D9), AESx(0x13F8F8EB), AESx(0xB398982B), AESx(0x33111122),
    AESx(0xBB6969D2), AESx(0x70D9D9A9), AESx(0x898E8E07), AESx(0xA7949433),
    AESx(0xB69B9B2D), AESx(0x221E1E3C), AESx(0x92878715), AESx(0x20E9E9C9),
    AESx(0x49CECE87), AESx(0xFF5555AA), AESx(0x78282850), AESx(0x7ADFDFA5),
    AESx(0x8F8C8C03), AESx(0xF8A1A159), AESx(0x80898909), AESx(0x170D0D1A),
    AESx(0xDABFBF65), AESx(0x31E6E6D7), AESx(0xC6424284), AESx(0xB86868D0),
    AESx(0xC3414182), AESx(0xB0999929), AESx(0x772D2D5A), AESx(0x110F0F1E),
    AESx(0xCBB0B07B), AESx(0xFC5454A8), AESx(0xD6BBBB6D), AESx(0x3A16162C)
  };
  
  __constant__ static const sph_u32 AES3_C[256] = {
    AESx(0xC6A56363), AESx(0xF8847C7C), AESx(0xEE997777), AESx(0xF68D7B7B),
    AESx(0xFF0DF2F2), AESx(0xD6BD6B6B), AESx(0xDEB16F6F), AESx(0x9154C5C5),
    AESx(0x60503030), AESx(0x02030101), AESx(0xCEA96767), AESx(0x567D2B2B),
    AESx(0xE719FEFE), AESx(0xB562D7D7), AESx(0x4DE6ABAB), AESx(0xEC9A7676),
    AESx(0x8F45CACA), AESx(0x1F9D8282), AESx(0x8940C9C9), AESx(0xFA877D7D),
    AESx(0xEF15FAFA), AESx(0xB2EB5959), AESx(0x8EC94747), AESx(0xFB0BF0F0),
    AESx(0x41ECADAD), AESx(0xB367D4D4), AESx(0x5FFDA2A2), AESx(0x45EAAFAF),
    AESx(0x23BF9C9C), AESx(0x53F7A4A4), AESx(0xE4967272), AESx(0x9B5BC0C0),
    AESx(0x75C2B7B7), AESx(0xE11CFDFD), AESx(0x3DAE9393), AESx(0x4C6A2626),
    AESx(0x6C5A3636), AESx(0x7E413F3F), AESx(0xF502F7F7), AESx(0x834FCCCC),
    AESx(0x685C3434), AESx(0x51F4A5A5), AESx(0xD134E5E5), AESx(0xF908F1F1),
    AESx(0xE2937171), AESx(0xAB73D8D8), AESx(0x62533131), AESx(0x2A3F1515),
    AESx(0x080C0404), AESx(0x9552C7C7), AESx(0x46652323), AESx(0x9D5EC3C3),
    AESx(0x30281818), AESx(0x37A19696), AESx(0x0A0F0505), AESx(0x2FB59A9A),
    AESx(0x0E090707), AESx(0x24361212), AESx(0x1B9B8080), AESx(0xDF3DE2E2),
    AESx(0xCD26EBEB), AESx(0x4E692727), AESx(0x7FCDB2B2), AESx(0xEA9F7575),
    AESx(0x121B0909), AESx(0x1D9E8383), AESx(0x58742C2C), AESx(0x342E1A1A),
    AESx(0x362D1B1B), AESx(0xDCB26E6E), AESx(0xB4EE5A5A), AESx(0x5BFBA0A0),
    AESx(0xA4F65252), AESx(0x764D3B3B), AESx(0xB761D6D6), AESx(0x7DCEB3B3),
    AESx(0x527B2929), AESx(0xDD3EE3E3), AESx(0x5E712F2F), AESx(0x13978484),
    AESx(0xA6F55353), AESx(0xB968D1D1), AESx(0x00000000), AESx(0xC12CEDED),
    AESx(0x40602020), AESx(0xE31FFCFC), AESx(0x79C8B1B1), AESx(0xB6ED5B5B),
    AESx(0xD4BE6A6A), AESx(0x8D46CBCB), AESx(0x67D9BEBE), AESx(0x724B3939),
    AESx(0x94DE4A4A), AESx(0x98D44C4C), AESx(0xB0E85858), AESx(0x854ACFCF),
    AESx(0xBB6BD0D0), AESx(0xC52AEFEF), AESx(0x4FE5AAAA), AESx(0xED16FBFB),
    AESx(0x86C54343), AESx(0x9AD74D4D), AESx(0x66553333), AESx(0x11948585),
    AESx(0x8ACF4545), AESx(0xE910F9F9), AESx(0x04060202), AESx(0xFE817F7F),
    AESx(0xA0F05050), AESx(0x78443C3C), AESx(0x25BA9F9F), AESx(0x4BE3A8A8),
    AESx(0xA2F35151), AESx(0x5DFEA3A3), AESx(0x80C04040), AESx(0x058A8F8F),
    AESx(0x3FAD9292), AESx(0x21BC9D9D), AESx(0x70483838), AESx(0xF104F5F5),
    AESx(0x63DFBCBC), AESx(0x77C1B6B6), AESx(0xAF75DADA), AESx(0x42632121),
    AESx(0x20301010), AESx(0xE51AFFFF), AESx(0xFD0EF3F3), AESx(0xBF6DD2D2),
    AESx(0x814CCDCD), AESx(0x18140C0C), AESx(0x26351313), AESx(0xC32FECEC),
    AESx(0xBEE15F5F), AESx(0x35A29797), AESx(0x88CC4444), AESx(0x2E391717),
    AESx(0x9357C4C4), AESx(0x55F2A7A7), AESx(0xFC827E7E), AESx(0x7A473D3D),
    AESx(0xC8AC6464), AESx(0xBAE75D5D), AESx(0x322B1919), AESx(0xE6957373),
    AESx(0xC0A06060), AESx(0x19988181), AESx(0x9ED14F4F), AESx(0xA37FDCDC),
    AESx(0x44662222), AESx(0x547E2A2A), AESx(0x3BAB9090), AESx(0x0B838888),
    AESx(0x8CCA4646), AESx(0xC729EEEE), AESx(0x6BD3B8B8), AESx(0x283C1414),
    AESx(0xA779DEDE), AESx(0xBCE25E5E), AESx(0x161D0B0B), AESx(0xAD76DBDB),
    AESx(0xDB3BE0E0), AESx(0x64563232), AESx(0x744E3A3A), AESx(0x141E0A0A),
    AESx(0x92DB4949), AESx(0x0C0A0606), AESx(0x486C2424), AESx(0xB8E45C5C),
    AESx(0x9F5DC2C2), AESx(0xBD6ED3D3), AESx(0x43EFACAC), AESx(0xC4A66262),
    AESx(0x39A89191), AESx(0x31A49595), AESx(0xD337E4E4), AESx(0xF28B7979),
    AESx(0xD532E7E7), AESx(0x8B43C8C8), AESx(0x6E593737), AESx(0xDAB76D6D),
    AESx(0x018C8D8D), AESx(0xB164D5D5), AESx(0x9CD24E4E), AESx(0x49E0A9A9),
    AESx(0xD8B46C6C), AESx(0xACFA5656), AESx(0xF307F4F4), AESx(0xCF25EAEA),
    AESx(0xCAAF6565), AESx(0xF48E7A7A), AESx(0x47E9AEAE), AESx(0x10180808),
    AESx(0x6FD5BABA), AESx(0xF0887878), AESx(0x4A6F2525), AESx(0x5C722E2E),
    AESx(0x38241C1C), AESx(0x57F1A6A6), AESx(0x73C7B4B4), AESx(0x9751C6C6),
    AESx(0xCB23E8E8), AESx(0xA17CDDDD), AESx(0xE89C7474), AESx(0x3E211F1F),
    AESx(0x96DD4B4B), AESx(0x61DCBDBD), AESx(0x0D868B8B), AESx(0x0F858A8A),
    AESx(0xE0907070), AESx(0x7C423E3E), AESx(0x71C4B5B5), AESx(0xCCAA6666),
    AESx(0x90D84848), AESx(0x06050303), AESx(0xF701F6F6), AESx(0x1C120E0E),
    AESx(0xC2A36161), AESx(0x6A5F3535), AESx(0xAEF95757), AESx(0x69D0B9B9),
    AESx(0x17918686), AESx(0x9958C1C1), AESx(0x3A271D1D), AESx(0x27B99E9E),
    AESx(0xD938E1E1), AESx(0xEB13F8F8), AESx(0x2BB39898), AESx(0x22331111),
    AESx(0xD2BB6969), AESx(0xA970D9D9), AESx(0x07898E8E), AESx(0x33A79494),
    AESx(0x2DB69B9B), AESx(0x3C221E1E), AESx(0x15928787), AESx(0xC920E9E9),
    AESx(0x8749CECE), AESx(0xAAFF5555), AESx(0x50782828), AESx(0xA57ADFDF),
    AESx(0x038F8C8C), AESx(0x59F8A1A1), AESx(0x09808989), AESx(0x1A170D0D),
    AESx(0x65DABFBF), AESx(0xD731E6E6), AESx(0x84C64242), AESx(0xD0B86868),
    AESx(0x82C34141), AESx(0x29B09999), AESx(0x5A772D2D), AESx(0x1E110F0F),
    AESx(0x7BCBB0B0), AESx(0xA8FC5454), AESx(0x6DD6BBBB), AESx(0x2C3A1616)
  };

  #define AES_ROUND_NOKEY(x0, x1, x2, x3)   do { \
    sph_u32 t0 = (x0); \
    sph_u32 t1 = (x1); \
    sph_u32 t2 = (x2); \
    sph_u32 t3 = (x3); \
    AES_ROUND_NOKEY_LE(t0, t1, t2, t3, x0, x1, x2, x3); \
  } while (0)

#define KEY_EXPAND_ELT(k0, k1, k2, k3)   do { \
    sph_u32 kt; \
    AES_ROUND_NOKEY(k1, k2, k3, k0); \
    kt = (k0); \
    (k0) = (k1); \
    (k1) = (k2); \
    (k2) = (k3); \
    (k3) = kt; \
  } while (0)

/*
 * This function assumes that "msg" is aligned for 32-bit access.
 */
#define c512(msg)  do { \
  sph_u32 p0, p1, p2, p3, p4, p5, p6, p7; \
  sph_u32 p8, p9, pA, pB, pC, pD, pE, pF; \
  sph_u32 x0, x1, x2, x3; \
  int r; \
 \
  p0 = h0; \
  p1 = h1; \
  p2 = h2; \
  p3 = h3; \
  p4 = h4; \
  p5 = h5; \
  p6 = h6; \
  p7 = h7; \
  p8 = h8; \
  p9 = h9; \
  pA = hA; \
  pB = hB; \
  pC = hC; \
  pD = hD; \
  pE = hE; \
  pF = hF; \
  /* round 0 */ \
  x0 = p4 ^ rk00; \
  x1 = p5 ^ rk01; \
  x2 = p6 ^ rk02; \
  x3 = p7 ^ rk03; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk04; \
  x1 ^= rk05; \
  x2 ^= rk06; \
  x3 ^= rk07; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk08; \
  x1 ^= rk09; \
  x2 ^= rk0A; \
  x3 ^= rk0B; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk0C; \
  x1 ^= rk0D; \
  x2 ^= rk0E; \
  x3 ^= rk0F; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  p0 ^= x0; \
  p1 ^= x1; \
  p2 ^= x2; \
  p3 ^= x3; \
  x0 = pC ^ rk10; \
  x1 = pD ^ rk11; \
  x2 = pE ^ rk12; \
  x3 = pF ^ rk13; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk14; \
  x1 ^= rk15; \
  x2 ^= rk16; \
  x3 ^= rk17; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk18; \
  x1 ^= rk19; \
  x2 ^= rk1A; \
  x3 ^= rk1B; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  x0 ^= rk1C; \
  x1 ^= rk1D; \
  x2 ^= rk1E; \
  x3 ^= rk1F; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  p8 ^= x0; \
  p9 ^= x1; \
  pA ^= x2; \
  pB ^= x3; \
 \
  for (r = 0; r < 3; r ++) { \
    /* round 1, 5, 9 */ \
    KEY_EXPAND_ELT(rk00, rk01, rk02, rk03); \
    rk00 ^= rk1C; \
    rk01 ^= rk1D; \
    rk02 ^= rk1E; \
    rk03 ^= rk1F; \
    if (r == 0) { \
      rk00 ^= sc_count0; \
      rk01 ^= sc_count1; \
      rk02 ^= sc_count2; \
      rk03 ^= SPH_T32(~sc_count3); \
    } \
    x0 = p0 ^ rk00; \
    x1 = p1 ^ rk01; \
    x2 = p2 ^ rk02; \
    x3 = p3 ^ rk03; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk04, rk05, rk06, rk07); \
    rk04 ^= rk00; \
    rk05 ^= rk01; \
    rk06 ^= rk02; \
    rk07 ^= rk03; \
    if (r == 1) { \
      rk04 ^= sc_count3; \
      rk05 ^= sc_count2; \
      rk06 ^= sc_count1; \
      rk07 ^= SPH_T32(~sc_count0); \
    } \
    x0 ^= rk04; \
    x1 ^= rk05; \
    x2 ^= rk06; \
    x3 ^= rk07; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk08, rk09, rk0A, rk0B); \
    rk08 ^= rk04; \
    rk09 ^= rk05; \
    rk0A ^= rk06; \
    rk0B ^= rk07; \
    x0 ^= rk08; \
    x1 ^= rk09; \
    x2 ^= rk0A; \
    x3 ^= rk0B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk0C, rk0D, rk0E, rk0F); \
    rk0C ^= rk08; \
    rk0D ^= rk09; \
    rk0E ^= rk0A; \
    rk0F ^= rk0B; \
    x0 ^= rk0C; \
    x1 ^= rk0D; \
    x2 ^= rk0E; \
    x3 ^= rk0F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    pC ^= x0; \
    pD ^= x1; \
    pE ^= x2; \
    pF ^= x3; \
    KEY_EXPAND_ELT(rk10, rk11, rk12, rk13); \
    rk10 ^= rk0C; \
    rk11 ^= rk0D; \
    rk12 ^= rk0E; \
    rk13 ^= rk0F; \
    x0 = p8 ^ rk10; \
    x1 = p9 ^ rk11; \
    x2 = pA ^ rk12; \
    x3 = pB ^ rk13; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk14, rk15, rk16, rk17); \
    rk14 ^= rk10; \
    rk15 ^= rk11; \
    rk16 ^= rk12; \
    rk17 ^= rk13; \
    x0 ^= rk14; \
    x1 ^= rk15; \
    x2 ^= rk16; \
    x3 ^= rk17; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk18, rk19, rk1A, rk1B); \
    rk18 ^= rk14; \
    rk19 ^= rk15; \
    rk1A ^= rk16; \
    rk1B ^= rk17; \
    x0 ^= rk18; \
    x1 ^= rk19; \
    x2 ^= rk1A; \
    x3 ^= rk1B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk1C, rk1D, rk1E, rk1F); \
    rk1C ^= rk18; \
    rk1D ^= rk19; \
    rk1E ^= rk1A; \
    rk1F ^= rk1B; \
    if (r == 2) { \
      rk1C ^= sc_count2; \
      rk1D ^= sc_count3; \
      rk1E ^= sc_count0; \
      rk1F ^= SPH_T32(~sc_count1); \
    } \
    x0 ^= rk1C; \
    x1 ^= rk1D; \
    x2 ^= rk1E; \
    x3 ^= rk1F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p4 ^= x0; \
    p5 ^= x1; \
    p6 ^= x2; \
    p7 ^= x3; \
    /* round 2, 6, 10 */ \
    rk00 ^= rk19; \
    x0 = pC ^ rk00; \
    rk01 ^= rk1A; \
    x1 = pD ^ rk01; \
    rk02 ^= rk1B; \
    x2 = pE ^ rk02; \
    rk03 ^= rk1C; \
    x3 = pF ^ rk03; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk04 ^= rk1D; \
    x0 ^= rk04; \
    rk05 ^= rk1E; \
    x1 ^= rk05; \
    rk06 ^= rk1F; \
    x2 ^= rk06; \
    rk07 ^= rk00; \
    x3 ^= rk07; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk08 ^= rk01; \
    x0 ^= rk08; \
    rk09 ^= rk02; \
    x1 ^= rk09; \
    rk0A ^= rk03; \
    x2 ^= rk0A; \
    rk0B ^= rk04; \
    x3 ^= rk0B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk0C ^= rk05; \
    x0 ^= rk0C; \
    rk0D ^= rk06; \
    x1 ^= rk0D; \
    rk0E ^= rk07; \
    x2 ^= rk0E; \
    rk0F ^= rk08; \
    x3 ^= rk0F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p8 ^= x0; \
    p9 ^= x1; \
    pA ^= x2; \
    pB ^= x3; \
    rk10 ^= rk09; \
    x0 = p4 ^ rk10; \
    rk11 ^= rk0A; \
    x1 = p5 ^ rk11; \
    rk12 ^= rk0B; \
    x2 = p6 ^ rk12; \
    rk13 ^= rk0C; \
    x3 = p7 ^ rk13; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk14 ^= rk0D; \
    x0 ^= rk14; \
    rk15 ^= rk0E; \
    x1 ^= rk15; \
    rk16 ^= rk0F; \
    x2 ^= rk16; \
    rk17 ^= rk10; \
    x3 ^= rk17; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk18 ^= rk11; \
    x0 ^= rk18; \
    rk19 ^= rk12; \
    x1 ^= rk19; \
    rk1A ^= rk13; \
    x2 ^= rk1A; \
    rk1B ^= rk14; \
    x3 ^= rk1B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk1C ^= rk15; \
    x0 ^= rk1C; \
    rk1D ^= rk16; \
    x1 ^= rk1D; \
    rk1E ^= rk17; \
    x2 ^= rk1E; \
    rk1F ^= rk18; \
    x3 ^= rk1F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p0 ^= x0; \
    p1 ^= x1; \
    p2 ^= x2; \
    p3 ^= x3; \
    /* round 3, 7, 11 */ \
    KEY_EXPAND_ELT(rk00, rk01, rk02, rk03); \
    rk00 ^= rk1C; \
    rk01 ^= rk1D; \
    rk02 ^= rk1E; \
    rk03 ^= rk1F; \
    x0 = p8 ^ rk00; \
    x1 = p9 ^ rk01; \
    x2 = pA ^ rk02; \
    x3 = pB ^ rk03; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk04, rk05, rk06, rk07); \
    rk04 ^= rk00; \
    rk05 ^= rk01; \
    rk06 ^= rk02; \
    rk07 ^= rk03; \
    x0 ^= rk04; \
    x1 ^= rk05; \
    x2 ^= rk06; \
    x3 ^= rk07; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk08, rk09, rk0A, rk0B); \
    rk08 ^= rk04; \
    rk09 ^= rk05; \
    rk0A ^= rk06; \
    rk0B ^= rk07; \
    x0 ^= rk08; \
    x1 ^= rk09; \
    x2 ^= rk0A; \
    x3 ^= rk0B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk0C, rk0D, rk0E, rk0F); \
    rk0C ^= rk08; \
    rk0D ^= rk09; \
    rk0E ^= rk0A; \
    rk0F ^= rk0B; \
    x0 ^= rk0C; \
    x1 ^= rk0D; \
    x2 ^= rk0E; \
    x3 ^= rk0F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p4 ^= x0; \
    p5 ^= x1; \
    p6 ^= x2; \
    p7 ^= x3; \
    KEY_EXPAND_ELT(rk10, rk11, rk12, rk13); \
    rk10 ^= rk0C; \
    rk11 ^= rk0D; \
    rk12 ^= rk0E; \
    rk13 ^= rk0F; \
    x0 = p0 ^ rk10; \
    x1 = p1 ^ rk11; \
    x2 = p2 ^ rk12; \
    x3 = p3 ^ rk13; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk14, rk15, rk16, rk17); \
    rk14 ^= rk10; \
    rk15 ^= rk11; \
    rk16 ^= rk12; \
    rk17 ^= rk13; \
    x0 ^= rk14; \
    x1 ^= rk15; \
    x2 ^= rk16; \
    x3 ^= rk17; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk18, rk19, rk1A, rk1B); \
    rk18 ^= rk14; \
    rk19 ^= rk15; \
    rk1A ^= rk16; \
    rk1B ^= rk17; \
    x0 ^= rk18; \
    x1 ^= rk19; \
    x2 ^= rk1A; \
    x3 ^= rk1B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    KEY_EXPAND_ELT(rk1C, rk1D, rk1E, rk1F); \
    rk1C ^= rk18; \
    rk1D ^= rk19; \
    rk1E ^= rk1A; \
    rk1F ^= rk1B; \
    x0 ^= rk1C; \
    x1 ^= rk1D; \
    x2 ^= rk1E; \
    x3 ^= rk1F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    pC ^= x0; \
    pD ^= x1; \
    pE ^= x2; \
    pF ^= x3; \
    /* round 4, 8, 12 */ \
    rk00 ^= rk19; \
    x0 = p4 ^ rk00; \
    rk01 ^= rk1A; \
    x1 = p5 ^ rk01; \
    rk02 ^= rk1B; \
    x2 = p6 ^ rk02; \
    rk03 ^= rk1C; \
    x3 = p7 ^ rk03; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk04 ^= rk1D; \
    x0 ^= rk04; \
    rk05 ^= rk1E; \
    x1 ^= rk05; \
    rk06 ^= rk1F; \
    x2 ^= rk06; \
    rk07 ^= rk00; \
    x3 ^= rk07; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk08 ^= rk01; \
    x0 ^= rk08; \
    rk09 ^= rk02; \
    x1 ^= rk09; \
    rk0A ^= rk03; \
    x2 ^= rk0A; \
    rk0B ^= rk04; \
    x3 ^= rk0B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk0C ^= rk05; \
    x0 ^= rk0C; \
    rk0D ^= rk06; \
    x1 ^= rk0D; \
    rk0E ^= rk07; \
    x2 ^= rk0E; \
    rk0F ^= rk08; \
    x3 ^= rk0F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p0 ^= x0; \
    p1 ^= x1; \
    p2 ^= x2; \
    p3 ^= x3; \
    rk10 ^= rk09; \
    x0 = pC ^ rk10; \
    rk11 ^= rk0A; \
    x1 = pD ^ rk11; \
    rk12 ^= rk0B; \
    x2 = pE ^ rk12; \
    rk13 ^= rk0C; \
    x3 = pF ^ rk13; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk14 ^= rk0D; \
    x0 ^= rk14; \
    rk15 ^= rk0E; \
    x1 ^= rk15; \
    rk16 ^= rk0F; \
    x2 ^= rk16; \
    rk17 ^= rk10; \
    x3 ^= rk17; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk18 ^= rk11; \
    x0 ^= rk18; \
    rk19 ^= rk12; \
    x1 ^= rk19; \
    rk1A ^= rk13; \
    x2 ^= rk1A; \
    rk1B ^= rk14; \
    x3 ^= rk1B; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    rk1C ^= rk15; \
    x0 ^= rk1C; \
    rk1D ^= rk16; \
    x1 ^= rk1D; \
    rk1E ^= rk17; \
    x2 ^= rk1E; \
    rk1F ^= rk18; \
    x3 ^= rk1F; \
    AES_ROUND_NOKEY(x0, x1, x2, x3); \
    p8 ^= x0; \
    p9 ^= x1; \
    pA ^= x2; \
    pB ^= x3; \
  } \
  /* round 13 */ \
  KEY_EXPAND_ELT(rk00, rk01, rk02, rk03); \
  rk00 ^= rk1C; \
  rk01 ^= rk1D; \
  rk02 ^= rk1E; \
  rk03 ^= rk1F; \
  x0 = p0 ^ rk00; \
  x1 = p1 ^ rk01; \
  x2 = p2 ^ rk02; \
  x3 = p3 ^ rk03; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk04, rk05, rk06, rk07); \
  rk04 ^= rk00; \
  rk05 ^= rk01; \
  rk06 ^= rk02; \
  rk07 ^= rk03; \
  x0 ^= rk04; \
  x1 ^= rk05; \
  x2 ^= rk06; \
  x3 ^= rk07; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk08, rk09, rk0A, rk0B); \
  rk08 ^= rk04; \
  rk09 ^= rk05; \
  rk0A ^= rk06; \
  rk0B ^= rk07; \
  x0 ^= rk08; \
  x1 ^= rk09; \
  x2 ^= rk0A; \
  x3 ^= rk0B; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk0C, rk0D, rk0E, rk0F); \
  rk0C ^= rk08; \
  rk0D ^= rk09; \
  rk0E ^= rk0A; \
  rk0F ^= rk0B; \
  x0 ^= rk0C; \
  x1 ^= rk0D; \
  x2 ^= rk0E; \
  x3 ^= rk0F; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  pC ^= x0; \
  pD ^= x1; \
  pE ^= x2; \
  pF ^= x3; \
  KEY_EXPAND_ELT(rk10, rk11, rk12, rk13); \
  rk10 ^= rk0C; \
  rk11 ^= rk0D; \
  rk12 ^= rk0E; \
  rk13 ^= rk0F; \
  x0 = p8 ^ rk10; \
  x1 = p9 ^ rk11; \
  x2 = pA ^ rk12; \
  x3 = pB ^ rk13; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk14, rk15, rk16, rk17); \
  rk14 ^= rk10; \
  rk15 ^= rk11; \
  rk16 ^= rk12; \
  rk17 ^= rk13; \
  x0 ^= rk14; \
  x1 ^= rk15; \
  x2 ^= rk16; \
  x3 ^= rk17; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk18, rk19, rk1A, rk1B); \
  rk18 ^= rk14 ^ sc_count1; \
  rk19 ^= rk15 ^ sc_count0; \
  rk1A ^= rk16 ^ sc_count3; \
  rk1B ^= rk17 ^ SPH_T32(~sc_count2); \
  x0 ^= rk18; \
  x1 ^= rk19; \
  x2 ^= rk1A; \
  x3 ^= rk1B; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  KEY_EXPAND_ELT(rk1C, rk1D, rk1E, rk1F); \
  rk1C ^= rk18; \
  rk1D ^= rk19; \
  rk1E ^= rk1A; \
  rk1F ^= rk1B; \
  x0 ^= rk1C; \
  x1 ^= rk1D; \
  x2 ^= rk1E; \
  x3 ^= rk1F; \
  AES_ROUND_NOKEY(x0, x1, x2, x3); \
  p4 ^= x0; \
  p5 ^= x1; \
  p6 ^= x2; \
  p7 ^= x3; \
  h0 ^= p8; \
  h1 ^= p9; \
  h2 ^= pA; \
  h3 ^= pB; \
  h4 ^= pC; \
  h5 ^= pD; \
  h6 ^= pE; \
  h7 ^= pF; \
  h8 ^= p0; \
  h9 ^= p1; \
  hA ^= p2; \
  hB ^= p3; \
  hC ^= p4; \
  hD ^= p5; \
  hE ^= p6; \
  hF ^= p7; \
  } while (0)

#define TPBS 256

__global__ __launch_bounds__(TPBS,2) 
void xevan_shavite512_gpu_hash_128(const uint32_t threads, uint64_t *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t r[16];
		uint64_t *Hash = &g_hash[thread<<3];

		__shared__ uint32_t AES0_S[256];
		__shared__ uint32_t AES1_S[256];
		__shared__ uint32_t AES2_S[256];
		//__shared__ uint32_t AES3_S[256];
        if(threadIdx.x < 256){
			uint32_t temp = AES0_C[threadIdx.x];
			AES0_S[threadIdx.x] = temp;
			AES1_S[threadIdx.x] = ROL8(temp);
			AES2_S[threadIdx.x] = ROL16(temp);
		}

		// fülle die Nachricht mit 64-byte (vorheriger Hash)
		*(uint2x4*)&r[ 0] = __ldg4((uint2x4*)&Hash[ 0]);
		*(uint2x4*)&r[ 8] = __ldg4((uint2x4*)&Hash[ 4]);

		//__syncthreads();

		sph_u32 h0 = SPH_C32(0x72FCCDD8), h1 = SPH_C32(0x79CA4727), h2 = SPH_C32(0x128A077B), h3 = SPH_C32(0x40D55AEC);
		sph_u32 h4 = SPH_C32(0xD1901A06), h5 = SPH_C32(0x430AE307), h6 = SPH_C32(0xB29F5CD1), h7 = SPH_C32(0xDF07FBFC);
		sph_u32 h8 = SPH_C32(0x8E45D73D), h9 = SPH_C32(0x681AB538), hA = SPH_C32(0xBDE86578), hB = SPH_C32(0xDD577E47);
		sph_u32 hC = SPH_C32(0xE275EADE), hD = SPH_C32(0x502D9FCD), hE = SPH_C32(0xB9357178), hF = SPH_C32(0x022A4B9A);

		// state
		sph_u32 rk00, rk01, rk02, rk03, rk04, rk05, rk06, rk07;
		sph_u32 rk08, rk09, rk0A, rk0B, rk0C, rk0D, rk0E, rk0F;
		sph_u32 rk10, rk11, rk12, rk13, rk14, rk15, rk16, rk17;
		sph_u32 rk18, rk19, rk1A, rk1B, rk1C, rk1D, rk1E, rk1F;

		sph_u32 sc_count0 = 0x400, sc_count1 = 0, sc_count2 = 0, sc_count3 = 0;

		rk00 = r[0];
		rk01 = r[1];
		rk02 = r[2];
		rk03 = r[3];
		rk04 = r[4];
		rk05 = r[5];
		rk06 = r[6];
		rk07 = r[7];
		rk08 = r[8];
		rk09 = r[9];
		rk0A = r[10];
		rk0B = r[11];
		rk0C = r[12];
		rk0D = r[13];
		rk0E = r[14];
		rk0F = r[15];
		rk10 = rk11 = rk12 = rk13 = rk14 = rk15 = rk16 = rk17 = 0;
		rk18 = rk19 = rk1A = rk1B = rk1C = rk1D = rk1E = rk1F = 0;

		__syncthreads();

		c512(anyarguments);

		rk00 = 0x80;
		rk01 = rk02 = rk03 = rk04 = rk05 = rk06 = rk07 = 0;
		rk08 = rk09 = rk0A = rk0B = rk0C = rk0D = rk0E = rk0F = 0;
		rk10 = rk11 = rk12 = rk13 = rk14 = rk15 = rk16 = rk17 = rk18 = rk19 = rk1A = 0;
		rk1B = 0x4000000;
		rk1C = rk1D = rk1E = 0;
		rk1F = 0x2000000;
		sc_count0 = 0;

		c512(anyarguments);

		r[0] = h0;
		r[1] = h1;
		r[2] = h2;
		r[3] = h3;
		r[4] = h4;
		r[5] = h5;
		r[6] = h6;
		r[7] = h7;
		r[8] = h8;
		r[9] = h9;
		r[10] = hA;
		r[11] = hB;
		r[12] = hC;
		r[13] = hD;
		r[14] = hE;
		r[15] = hF;

		*((uint2x4*)&Hash[ 0]) = *(uint2x4*)&r[ 0];
		*((uint2x4*)&Hash[ 4]) = *(uint2x4*)&r[ 8];
	}
}

__host__
void xevan_shavite512_cpu_hash_128(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + TPBS-1)/TPBS);
	dim3 block(TPBS);

	xevan_shavite512_gpu_hash_128<<<grid, block>>>(threads, (uint64_t*)d_hash);
}
