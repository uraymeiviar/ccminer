# ccminer

forked from suprminer by [ocminer](https://github.com/ocminer/suprminer) 

original ccminer project is based on Christian Buchner's &amp; Christian H.'s CUDA project, no more active on github since 2014, since then it actively developed by [tpruvot](https://github.com/tpruvot/ccminer)

BTC donation address: `1AJdfCpLWPNoAMDfHF1wD5y8VgKSSTHxPo` (tpruvot)

A part of the recent algos were originally written by [djm34](https://github.com/djm34) and [alexis78](https://github.com/alexis78)

RVN donation address: `RFg6gtkg6fFzsY64hWANTQC79yvNaQv8sL` (alexis)

Compile on Linux
----------------

for ubuntu based :
`sudo apt install build-essential libcurl4-gnutls-dev openssl-dev`

then install nvidia cuda toolkit from official [website](https://developer.nvidia.com/cuda-toolkit), follow the instruction there, and then just do `./build.sh` inside project directory, make sure all cuda toolkit development tools and library are in your environment variable paths (default to `/usr/local/cuda`), make sure you can execute `gcc` and `nvcc`

in case you need to customize the build process, look at `Makefile.am`,`build.sh`,`configure.sh`,`configure.ac`


Supported Algorithm
-------------------
    bastion     Hefty bastion
    bcd         BitcoinDiamond
    bitcore     Timetravel-10
    blake       Blake 256 (SFR)
    blake2s     Blake2-S 256 (NEVA)
    blakecoin   Fast Blake 256 (8 rounds)
    bmw         BMW 256
    cryptolight AEON cryptonight (MEM/2)
    cryptonight XMR cryptonight
    c11/flax    X11 variant
    decred      Decred Blake256
    deep        Deepcoin
    equihash    Zcash Equihash
    dmd-gr      Diamond-Groestl
    fresh       Freshcoin (shavite 80)
    fugue256    Fuguecoin
    graft       Cryptonight v8\n\
    groestl     Groestlcoin\n"
    heavy       Heavycoin\n"
    hmq1725     Doubloons / Espers
    hsr         X13+SM3
    jackpot     JHA v8
    keccak      Deprecated Keccak-256
    keccakc     Keccak-256 (CreativeCoin)
    lbry        LBRY Credits (Sha/Ripemd)
    luffa       Joincoin
    lyra2       CryptoCoin
    lyra2v2     VertCoin
    lyra2z      ZeroCoin (3rd impl)
    myr-gr      Myriad-Groestl
    monero      XMR cryptonight (v7)\n\
    neoscrypt   FeatherCoin, Phoenix, UFO...
    nist5       NIST5 (TalkCoin)
    penta       Pentablake hash (5x Blake 512)
    phi         BHCoin
    polytimos   Politimos
    quark       Quark
    qubit       Qubit
    sha256d     SHA256d (bitcoin)
    sha256t     SHA256 x3
    sia         SIA (Blake2B)
    sib         Sibcoin (X11+Streebog)
    scrypt      Scrypt
    scrypt-jane Scrypt-jane Chacha
    skein       Skein SHA2 (Skeincoin)
    skein2      Double Skein (Woodcoin)
    skunk       Skein Cube Fugue Streebog
    stellite    Cryptonight v3\n\
    s3          S3 (1Coin)
    timetravel  Machinecoin permuted x8
    tribus      Denarius
    vanilla     Blake256-8 (VNL)
    veltor      Thorsriddle streebog
    whirlcoin   Old Whirlcoin (Whirlpool algo)
    whirlpool   Whirlpool algo
    x11evo      Permuted x11 (Revolver)
    x11         X11 (DarkCoin)
    x12         X12 (GalaxyCash)\n\
    x13         X13 (MaruCoin)
    x14         X14
    x15         X15
    x16r        X16R (Raven)
    x16s	    X16S (Pidgeon)
    x17         X17
    wildkeccak  Boolberry
    zr5         ZR5 (ZiftrCoin)

run with argument `--help` for more information to use the program