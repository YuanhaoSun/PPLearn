def SSK(lamb, p):
    """Return subsequence kernel"""
    def SSKernel(xi,xj,lamb,p):
        mykey = (xi, xj) if xi>xj else (xj, xi)
        if not mykey in cache:
            dps = []
            for i in xrange(len(xi)):
                dps.append([lamb**2 if xi[i] == xj[j] else 0 for j in xrange(len(xj))])
            dp = []
            for i in xrange(len(xi)+1):
                dp.append([0]*(len(xj)+1))
            k = [0]*(p+1)
            for l in xrange(2, p + 1):
                for i in xrange(len(xi)):
                    for j in xrange(len(xj)):
                        dp[i+1][j+1] = dps[i][j] + lamb * dp[i][j+1] + lamb * dp[i+1][j] - lamb**2 * dp[i][j]
                        if xi[i] == xj[j]:
                            dps[i][j] = lamb**2 * dp[i][j]
                            k[l] = k[l] + dps[i][j]
            cache[mykey] = k[p]
    return cache[mykey]
return lambda xi, xj: SSKernel(xi,xj,lamb,p)/(SSKernel(xi,xi,lamb,p) * SSKernel(xj,xj,lamb,p))**0.5