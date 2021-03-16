// host
// .type c,@object
// .local c
// .comm c,512,16
// .addrsig_sym c
// device
// .visible .const .align #alignment .b8 c[512];
// mov.u64 %rd1, c;
// cvta.const.u64 %rd2, %rd1;
// or
// ld.const.u64 %rd1, [c+i];
#[constant] static c: [f64; 64];

