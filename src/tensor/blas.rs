#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    pub fn cblas_sgemm(
        layout: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(target_os = "macos")]
pub const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
pub const CBLAS_NO_TRANS: i32 = 111;
#[cfg(target_os = "macos")]
pub const CBLAS_TRANS: i32 = 112;

/// General matrix multiplication: C = alpha * op(A) * op(B) + beta * C
/// - If `trans_a` is false: A has shape (m, k)
/// - If `trans_a` is true: A has shape (k, m) in memory and is transposed to (m, k)
/// - If `trans_b` is false: B has shape (k, n)
/// - If `trans_b` is true: B has shape (n, k) in memory and is transposed to (k, n)
/// - C has shape (m, n)
#[inline(always)]
pub unsafe fn gemm(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    #[cfg(target_os = "macos")]
    {
        let ta = if trans_a { CBLAS_TRANS } else { CBLAS_NO_TRANS };
        let tb = if trans_b { CBLAS_TRANS } else { CBLAS_NO_TRANS };
        let lda = if trans_a { m as i32 } else { k as i32 };
        let ldb = if trans_b { k as i32 } else { n as i32 };
        let ldc = n as i32;

        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                ta,
                tb,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let (rsa, csa) = if trans_a { (1, m as isize) } else { (k as isize, 1) };
        let (rsb, csb) = if trans_b { (1, k as isize) } else { (n as isize, 1) };
        let (rsc, csc) = (n as isize, 1);

        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                alpha,
                a,
                rsa,
                csa,
                b,
                rsb,
                csb,
                beta,
                c,
                rsc,
                csc,
            );
        }
    }
}
