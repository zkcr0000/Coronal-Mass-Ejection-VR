#! /usr/bin/env python3

import os
import sys
import numpy
import math
import random
import struct
from functools import reduce
### import scipy.spatial  ## only needed if writevolumebgeo()

### from trajio import TrajIO  ## only needed if writetraj()

def ptr2xyz( phi, theta, r ):
    """ptr2xyz(): converts three numpy vectors of phi, theta, r coordinates to (x,y,z) cartesian form, returning a single Nx3 array"""

    xyz = numpy.ndarray( ( len(phi), 3 ) )
    sintheta = numpy.sin( theta )
    # x = r sin theta cos phi
    # y = r sin theta sin phi
    # z = r cos theta
    xyz[:,0] = r * sintheta * numpy.cos( phi )
    xyz[:,1] = r * sintheta * numpy.sin( phi )
    xyz[:,2] = r * numpy.cos( theta )
    return xyz

def ptrvex2xyz( phi, theta, r, ptrvex ):
    """ptrvex2xyz( phi,theta,r, [ [vphi0,vtheta0,vr0], [vphi1,vtheta1,vr1], ... ] )
        Takes three numpy arrays of phi, theta, r coordinates defining a sequence of points,
        plus one or more lists of phi,theta,r direction vectors based at those same points.
        Returns: something in similar structure, but with each phi,theta,r replaced by a single Nx3 numpy array.
        E.g.:
           xyz, vec0, vec1 = ptrvex2xyz( phi,theta,r, [ [vphi0,vtheta0,vr0], [vphi1,vtheta1,vr1] ] )
           where xyz, vec0, and vec1 are each Nx3 arrays."""

 
    # assume without checking that len(phi) = len(theta) = len(r) = len(ptrvex[0][0]) etc.   
    
    xyz = numpy.ndarray( ( len(phi), 3 ) )

    sinphi = numpy.sin( phi )
    cosphi = numpy.cos( phi )
    sintheta = numpy.sin( theta )
    costheta = numpy.cos( theta )

    # x = r sin theta cos phi
    # y = r sin theta sin phi
    # z = r cos theta
    xyz[:,0] = r * sintheta * cosphi
    xyz[:,1] = r * sintheta * sinphi
    xyz[:,2] = r * costheta

    xyzvex = [ xyz ]

    # basis - it's just the partial derivatives of position with respect to phi, theta, r:
    # phi^ =   { -sintheta*sinphi, sintheta*cosphi, 0 }
    # theta^ = { costheta*cosphi, costheta*sinphi, -sintheta }
    # r^ =     { sintheta*cosphi, sintheta*sinphi, costheta }

    
    xbasis = numpy.array( [ -sintheta*sinphi,
                             costheta*cosphi,
                             sintheta*cosphi ] )
    ybasis = numpy.array( [ sintheta*cosphi,
                            costheta*sinphi,
                            sintheta*sinphi ] )
    zbasis = numpy.array( [ numpy.zeros( len(phi) ),
                            -sintheta,
                             costheta ] )
    basis = numpy.array( [ xbasis, ybasis, zbasis ] )

    for vphi, vtheta, vr in ptrvex:
        # again assume without checking that len(vphi) = len(vtheta) = len(vr) = len(phi) = etc.
        ## vxyz[i,k] = sum over j of   basis[k,j,i] * vptr[j,i]
        # Can't we do this with tensordot() somehow?
        vxyz = numpy.ndarray( ( len(phi), 3 ) )
        for k in range(3):
            vxyz[:,k] = basis[k,0,:]*vphi + basis[k,1,:]*vtheta + basis[k,2,:]*vr
        
        #vvxyz = numpy.ndarray( ( len(phi), 3 ) )
        #for i in range(len(phi)):
        #    vvxyz[i, 0] = basis[0,0,i]*vphi[i] + basis[0,1,i]*vtheta[i] + basis[0,2,i]*vr[i]
        #    vvxyz[i, 1] = basis[1,0,i]*vphi[i] + basis[1,1,i]*vtheta[i] + basis[1,2,i]*vr[i]
        #    vvxyz[i, 2] = basis[2,0,i]*vphi[i] + basis[2,1,i]*vtheta[i] + basis[2,2,i]*vr[i]
        #if vvxyz != vxyz:
        #    raise Exception("Trouble, vvxyz != vxyz")

        xyzvex.append( vxyz )

    return xyzvex

class YFan:

    try:
        # if /proc/meminfo exists, assume we're on a memory-rich linux machine.  ha.
        memf = open('/proc/meminfo', 'r')
        memf.close()
        readFields = True
    except:
        readFields = False

    useNearest = False

    def __init__(self, yfname):
        """ YFan(filename)
        Opens and reads one of Yuhong Fan's data files, containing magnetic-field-lines plus grids
        """

        self.yf = open(yfname, 'rb')

        # Use filename hack to determine file format
        head, tail = os.path.split( yfname )
        # "datafdls_NNN.dat" => original file format, with Br,theta,phi, Vr,theta,phi, density, temperature, and no fieldline ids
        # "bdatafdls_NNN.dat" => July2014 file format, with Br,theta,phi, and with fieldline ids
        self.fmtjul2014 = (tail[0] == 'b')

        # set switches according to what that format implies
        readBs = self.readFields
        readVRTs = self.readFields and not self.fmtjul2014
        skipVRTs = not self.readFields and not self.fmtjul2014
        readLineIds = self.readFields and self.fmtjul2014

        # yfan files are raw binary, 32-bit little-endian floats and int
        # the grid is a section of a spherical shell, expressed in (r, theta, phi) coordinates
        # the field lines are an array

        # Header format:
        #  "##" marks the data entries, in the order that they appear as you read the data file
        ## float time
        ## int nr   # grid size in r (radial) direction
        ## int nth  # grid size in theta (latitude) direction
        ## int nph  # grid size in phi (longitude) direction
        ## int nfdl # number of magnetic-field lines
        ## int maxpts # number of points in longest magnetic-field line
        self.time, = self.readN( 1 )
        self.nr, self.nth, self.nph, self.nfdl, self.maxpts = self.readN( 5, numpy.int32 )

        # coordinate grid
        # the 3-D grid is regularly connected in each of r, theta, and phi
        # but not necessarily with uniform spacing,
        # so the rCoord[] array gives the radial position of each slice of the grid, etc.

        ## float rCoord[ nr ]	# r coordinate
        ## float thCoord[ nth ]  # theta coordinate
        ## float phCoord[ nph ]	# phi coordinate
        self.rCoord = self.readN( self.nr )
        self.thCoord = self.readN( self.nth )
        self.phCoord = self.readN( self.nph )

        # IDL arrays are Fortran-style: first index varies fastest -- Br(r,th,ph)
        # Python arrays are C-style: last index variest fastest -- Br[ph, th, r]

        # python 'shape' vector for all 3D grid variables - (nph, nth, nr)
        # NOTE: in storage order, phi varies slowest, r varies fastest
        rthphi = (self.nph, self.nth, self.nr)

        if readBs:
            # B (magnetic) field is a 3-component vector at each grid point,
            # In the file, it's stored as three successive nph * nth * nr grids of floats,
            # one grid for each vector component

            ## float Br[ nph, nth, nr ]
            ## float Bth[ nph, nth, nr ]
            ## float Bph[ nph, nth, nr ]

            self.Br = self.readN( rthphi )
            self.Bth = self.readN( rthphi )
            self.Bphi = self.readN( rthphi )
        else:
            self.skipN( rthphi )
            self.skipN( rthphi )
            self.skipN( rthphi )
            self.Br = self.Bth = self.Bphi = None

        if readVRTs:
            # velocity field, organized the same way - a 3D vector at each grid point,
            # stored as three successive nph*nth*nr grids of floats

            ## float Vr[ nph, nth, nr ]
            ## float Vth[ nph, nth, nr ]
            ## float Vph[ nph, nth, nr ]
            self.Vr = self.readN( rthphi )
            self.Vth = self.readN( rthphi )
            self.Vphi = self.readN( rthphi )

            # density and temperature are scalar grids

            ## float density[ nph, nth, nr ]
            ## float temperature[ nph, nth, nr ]
            self.density = self.readN( rthphi )
            self.temperature = self.readN( rthphi )

        else:
            if skipVRTs:
                for i in range(5):  # Vr, Vth, Vph, density, temperature
                    self.skipN( rthphi )

            self.Vr = self.Vth = self.Vphi = None
            self.density = self.temperature = None

        # Magnetic field line curves
        # the lines don't all have the same number of points along them,
        # but for simplicity, she writes the lines as a regular 2D array
        # indexed as [ number_of_field_lines, max_points_per_field_line ]
        
        ## int npoints[ nfdl ]  # number of actual points along each field line
        ## float color[ nfdl ]  # color assigned to each field line (constant all along that line)
        self.npoints = self.readN( self.nfdl, numpy.int32 )
        self.color = self.readN( self.nfdl )
        if readLineIds:
            self.lineids = self.readN( self.nfdl, numpy.int32 )
        else:
            self.lineids = numpy.array( xrange(self.nfdl) )

        ## float fdlr[ nfdl, maxpts ]  # r coordinate of each point of each field line
        ## float fdlth[ nfdl, maxpts ]  # theta coordinate of each point of each field line
        ## float fdlph[ nfdl, maxpts ]  # phi coordinate of each point of each field line
        ## float fdltemp[ nfdl, maxpts ]  # temperature at each point of each field line

        fldsshape = (self.nfdl, self.maxpts)
        self.fdlr = self.readN( fldsshape )
        self.fdlth = self.readN( fldsshape )
        self.fdlphi = self.readN( fldsshape )
        self.fdltemp = self.readN( fldsshape )   # temperature along field line

        self.rsol = self.readN(1)

        self.samplelines()

    # takes a 1-D vector of M coordinate values and a 1-D "scale" variable (coordinates at each grid cell position, length N)
    # returns a vector of M indices, each in range 0 .. N-1, corresponding to the (round-to-nearest? floor?) grid cell for that input value
    def coord2ix( self, cvec, scalevec ):

        # Neat: searchsorted() finds scalevec[] index where each cvec[] entry belongs
        crawix = scalevec.searchsorted( cvec )

        N = len(scalevec)
        # crawix[] entries are in 0 .. N
        cix = numpy.fmin( crawix, N-2 )

        # round-to-nearest grid point.
        # I.e. round up where cvec[] exceeds midpoint of its interval.
        # Could go further to produce an interpolation fraction too.

        # nearest integer cell:
        midpoint = 0.5 * (scalevec[ 1 : N-1 ] + scalevec[ 0 : N-2 ])
        if self.useNearest:
            cix += ( cvec > midpoint[ cix ] )
        else:
            span = (scalevec[ 1 : N-1 ] - scalevec[ 0 : N-2 ])
            invspan = 1.0 / (span + (span==0))
            print("shapes: cvec", cvec.shape, "cix", cix.shape, "midpoint", midpoint.shape, "span", span.shape, file=sys.stderr)
            dcix = numpy.clip( (cvec - midpoint[ cix ]) * invspan[ cix ], -0.5, 0.5 )
            cix += ( 0.5 + dcix ).astype( int )

        return cix

    # trilinear interpolation
    def sampleat( self, v, ix0, ix1, ix2 ):

        if v is None:
            return None

        # Note we use floor() and ceil() rather than floor() and floor()+1.
        # This allows index to safely range from [ 0 .. N-1 ] inclusive
        i0a = numpy.floor( ix0 ).astype( numpy.int32 )
        i0b = numpy.ceil( ix0 ).astype( numpy.int32 )
        w0b = ix0 - i0a;  w0a = 1-w0b

        i1a = numpy.floor( ix1 ).astype( numpy.int32 )
        i1b = numpy.ceil( ix1 ).astype( numpy.int32 )
        w1b = ix1 - i1a;  w1a = 1-w1b

        i2a = numpy.floor( ix2 ).astype( numpy.int32 )
        i2b = numpy.ceil( ix2 ).astype( numpy.int32 )
        w2b = ix2 - i2a;  w2a = 1-w2b

        
        # return v[ i0a,i1a,i2a ]

        vaa_ = v[ i0a,i1a,i2a ]*w2a + v[ i0a,i1a,i2b ]*w2b
        vab_ = v[ i0a,i1b,i2a ]*w2a + v[ i0a,i1b,i2b ]*w2b
        vba_ = v[ i0b,i1a,i2a ]*w2a + v[ i0b,i1a,i2b ]*w2b
        vbb_ = v[ i0b,i1b,i2a ]*w2a + v[ i0b,i1b,i2b ]*w2b

        va__ = vaa_*w1a + vab_*w1b
        vb__ = vba_*w1a + vbb_*w1b

        return va__*w0a + vb__*w0b

    # sample 3D-field-variables (if available) along magnetic field curves
    def samplelines( self ):

        self.fdl_ir = self.coord2ix( self.fdlr, self.rCoord )
        self.fdl_ith = self.coord2ix( self.fdlth, self.thCoord )
        self.fdl_iphi = self.coord2ix( self.fdlphi, self.phCoord )

        self.fdlBr = self.sampleat( self.Br, self.fdl_iphi, self.fdl_ith, self.fdl_ir )
        self.fdlBth = self.sampleat( self.Bth, self.fdl_iphi, self.fdl_ith, self.fdl_ir )
        self.fdlBphi = self.sampleat( self.Bphi, self.fdl_iphi, self.fdl_ith, self.fdl_ir )

        self.fdlVr = self.sampleat( self.Vr, self.fdl_iphi, self.fdl_ith, self.fdl_ir )
        self.fdlVth = self.sampleat( self.Vth, self.fdl_iphi, self.fdl_ith, self.fdl_ir )
        self.fdlVphi = self.sampleat( self.Vphi, self.fdl_iphi, self.fdl_ith, self.fdl_ir )

        self.fdldensity = self.sampleat( self.density, self.fdl_iphi, self.fdl_ith, self.fdl_ir )

        

    # read from file
    # shape: an integer, or tuple/list giving array shape
    def readN( self, shape, dtype=numpy.float32 ):
        if hasattr(shape, "__iter__"):
            count = reduce( lambda a,b: a*b, shape, 1 )
        else:
            count = shape
        sz = numpy.dtype( dtype ).itemsize
        # print("readN", dtype, count, shape, file=sys.stderr)
        bbytes = self.yf.read( count * sz )
        return numpy.reshape( numpy.frombuffer( bbytes, dtype ), shape )

    def skipN( self, shape, dtype=numpy.float32 ):
        if hasattr(shape, "__iter__"):
            count = reduce( lambda a,b: a*b, shape, 1 )
        else:
            count = shape
        sz = numpy.dtype( dtype ).itemsize
        print("skipN", dtype, count, shape, file=sys.stderr)
        self.yf.seek( count * sz, 1 )

    def writetraj( self, outf, cartesian, want = "Bvec Bmag Vvec Vmag", box=(0.5,0.5,0.5) ):

        import trajio

        traj = TrajIO(outf, 'w')

        attrnames = ["temperature", "density"]
        itemp = 0
        idensity = 1
        iBvec = iBmag = iVvec = iVmag = None
        if "Bvec" in want:
            iBvec = len(attrnames)
            attrnames.extend( cartesian and ("Bx","By","Bz") or ("Br", "Btheta", "Bphi") )
        if "Bmag" in want:
            iBmag = len(attrnames)
            attrnames.append( "Bmag" )
        if "Vvec" in want:
            iVvec = len(attrnames)
            attrnames.extend( cartesian and ("Vx","Vy","Vz") or ("Vr", "Vtheta", "Vphi") )
        if "Vmag" in want:
            iVmag = len(attrnames)
            attrnames.append( "Vmag" )

        nattr = len(attrnames)

        trajlist = []

        for i in range(self.nfdl):
            nk = self.npoints[i]

            phi,theta,r = self.fdlphi[i][0:nk], self.fdlth[i][0:nk], self.fdlr[i][0:nk]
            temp = self.fdltemp[i][0:nk]

            trajdata = numpy.ndarray( (nk, 3+nattr) )

            if cartesian:
                trajdata[:, 0:3] = ptr2xyz( phi, theta, r )
            else:
                trajdata[:, 0] = phi
                trajdata[:, 1] = theta
                trajdata[:, 2] = r

            trajdata[:, 3+itemp] = temp

            if self.fdldensity is not None:
                br,btheta,bphi = self.fdlBr[i][0:nk], self.fdlBth[i][0:nk], self.fdlBphi[i][0:nk]

                if iBmag is not None:
                    bmag = numpy.sqrt( br*br + btheta*btheta + bphi*bphi )
                    trajdata[:, 3+iBmag] = bmag
                if iVmag is not None:
                    speed = numpy.sqrt( vr*vr + vtheta*vtheta + vphi*vphi )
                    trajdata[:, 3+iVmag] = speed

                if cartesian:
                    xyz, bxyz, vxyz = ptrvex2xyz( phi, theta, r,  [ [bphi,btheta,br], [vphi,vtheta,vr] ] )
                    trajdata[:, 0:3] = xyz
                    if iBvec is not None:
                        trajdata[:, 3+iBvec+0:3+iBvec+3] = bxyz
                    if iVvec is not None:
                        trajdata[:, 3+iVvec+0:3+iVvec+3] = vxyz
                else:
                    if iBvec is not None:
                        trajdata[:, 3+iBvec+0] = bphi
                        trajdata[:, 3+iBvec+1] = btheta
                        trajdata[:, 3+iBvec+2] = br
                    if iVvec is not None:
                        trajdata[:, 3+iVvec+0] = vphi
                        trajdata[:, 3+iVvec+1] = vtheta
                        trajdata[:, 3+iVvec+2] = vr

            # Build an entry for this trajectory, as needed by TrajIO.writetraj().
            # each entry expects a tuple:  trajid, startdatatime, trajdata[npoints][3+nattr]
            # We take over the useless 'startdatatime' and replace it with the integer color index.
            trajlist.append( ( i, self.color[i], trajdata ) )


        # send them all to the traj file.
        traj.writetraj( attrnames, trajlist )
            
                
    def writebspeck( self, outf, cinfo ):
        cx,cy,cz,cr = cinfo
        # cartesian assumed true
        datavars = [ "Bx", "By", "Bz" ]

        if isinstance(outf, str):
            outf = open(outf, 'w')
        
        print("datavar 0 Bx", file=outf)
        print("datavar 1 By", file=outf)
        print("datavar 2 Bz", file=outf)
        print("datavar 3 Br", file=outf)
        print("datavar 4 Bth", file=outf)
        print("datavar 5 Bphi", file=outf)

        every = 2
        phiN, thN, rN = self.phiCoord[::every], self.thCoord[::every], self.rCoord[::every]
        for r in rN:
            gridphi, gridth = numpy.meshgrid( phiN, thN )
            
        px,py,pz = ptr2xyz( self.phiCoord, self.thCoord, self.rCoord )

    def writespeck( self, outf, cartesian, currentnorm ):

        datavars = [ "temp", "color", "id" ]
        format = "%g %g %g %g %d %d"
        if self.fdlBr is not None:
            if cartesian:
                datavars.extend( ["Bx", "By", "Bz", "Bmag"] )
            else:
                datavars.extend( ["Bphi", "Btheta", "Br", "Bmag"] )
            datavars.extend( ["currentnorm"] )
            format += " %g %g %g %g %g"

        if self.fdldensity is not None:
            if cartesian:
                datavars.extend( ["Vx","Vy","Vz","speed"] )
            else:
                datavars.extend( ["Vphi", "Vtheta", "Vr", "speed"] )
            datavars.extend( ["density"] )
            format += " %g %g %g %g %g"

        # value of one row of particle data
        val = [0] * (3 + len(datavars))

        if isinstance(outf, str):
            outf = open(outf, 'w')

        print("#! /usr/bin/env partiview\n", file=outf)

        print("## ", " ".join(sys.argv), file=outf)

        # use radialnorm[] parameters to make a radial scaling function (array)
        rscaling = self.radialfunc( currentnorm )

        # for testing, report the range of radial scale factors as a comment in the speck file - innermost, mid, outermost
        rmid = int(self.nr/2)
        print("## radialnorm: %s => r[0]%g => %g;   r[%d]%g => %g;   r[%d]%g => %g" % \
              ( str(currentnorm),			\
                self.rCoord[0], rscaling[0],		\
                rmid, self.rCoord[rmid], rscaling[rmid], \
                self.nr-1, self.rCoord[-1], rscaling[-1] ), \
             file=outf)

        # Apply radial norm to current field and sample at lines
        normcurrent = self.current( 1.0 / rscaling )
        linecurrent = self.sampleat( normcurrent, self.fdl_iphi, self.fdl_ith, self.fdl_ir )

        print("", file=outf)
        for i in range(len(datavars)):
            print("datavar %d %s" % (i, datavars[i]), file=outf)

        print("\n", file=outf)

        for i in range(self.nfdl):

            nk = self.npoints[i]

            phi,theta,r = self.fdlphi[i][0:nk], self.fdlth[i][0:nk], self.fdlr[i][0:nk]
            temp = self.fdltemp[i][0:nk]


            if self.fdlBr is not None:
                bphi,btheta,br = self.fdlBphi[i][0:nk], self.fdlBth[i][0:nk], self.fdlBr[i][0:nk]
                bmag = numpy.sqrt( br*br + btheta*btheta + bphi*bphi )


            if self.fdldensity is not None:
                vphi,vtheta,vr = self.fdlVphi[i][0:nk], self.fdlVth[i][0:nk], self.fdlVr[i][0:nk]
                speed = numpy.sqrt( vr*vr + vtheta*vtheta + vphi*vphi )

                density = self.fdldensity[i][0:nk]

            color = self.color[i]
            id = self.lineids[i]


            if self.fdlBr is not None and self.fdldensity is not None:
                if cartesian:
                    xyz, bxyz, vxyz = ptrvex2xyz( phi, theta, r,  [ [bphi,btheta,br], [vphi,vtheta,vr] ] )
                    for k in range(nk):
                        val[0:3] = xyz[k]
                        val[3:6] = temp[k], color, id
                        val[6:9] = bxyz[k]
                        val[9] = bmag[k]
                        val[10] = linecurrent[i,k]
                        val[11:14] = vxyz[k]
                        val[14] = speed[k]
                        val[15] = density[k]
                        print(format % tuple(val), file=outf)
                else:
                    for k in range(nk):
                        val[0:3] = phi[k], theta[k], r[k]
                        val[3:6] = temp[k], color, id
                        val[6:9] = bphi[k], btheta[k], br[k]
                        val[9] = bmag[k]
                        val[10] = linecurrent[i,k]
                        val[11:14] = bphi[k], btheta[k], br[k]
                        val[14] = speed[k]
                        val[15] = density[k]
                        print(format % tuple(val), file=outf)

            elif self.fdlBr is not None:
                if cartesian:
                    xyz, bxyz = ptrvex2xyz( phi, theta, r,  [ [bphi,btheta,br] ] )
                    for k in range(nk):
                        val[0:3] = xyz[k]
                        val[3:6] = temp[k], color, id
                        val[6:9] = bxyz[k]
                        val[9] = bmag[k]
                        val[10] = linecurrent[i,k]
                        print(format % tuple(val), file=outf)
                else:
                    for k in range(nk):
                        val[0:3] = phi[k], theta[k], r[k]
                        val[3:6] = temp[k], color, id
                        val[6:9] = bphi[k], btheta[k], br[k]
                        val[9] = bmag[k]
                        val[10] = linecurrent[i,k]
                        print(format % tuple(val), file=outf)

            else:
                if cartesian:
                    xyz = ptr2xyz( phi, theta, r )
                    for k in range(nk):
                        val[0:3] = xyz[k]
                        val[3:6] = temp[k], color, id
                        print(format % tuple(val), file=outf)
                else:
                    for k in range(nk):
                        val[0:3] = phi[k], theta[k], r[k]
                        val[3:6] = temp[k], color, id
                        print(format % tuple(val), file=outf)


    def writegeo(self, outf, cartesian):
        """Takes the yfan data and writes a .GEO file to be read by Houdini"""
        
        outfile = open(outf, 'w+')
        
        # Count the total number of points on all of the field lines. 
        npoints = 0
        for i in range(self.nfdl):
            nk = self.npoints[i]
            npoints += nk
            
        # .GEO header
        print("PGEOMETRY V2", file=outfile)
        print("NPoints " + str(npoints) + " NPrims " + str(self.nfdl), file=outfile)
        print("NPointGroups 0 NPrimGroups 0", file=outfile)
        print("NPointAttrib 5 NVertexAttrib 0 NPrimAttrib 1 NAttrib 7", file=outfile)

        # Point attribute definition
        # These attributes are for each point
        print("PointAttrib", file=outfile)
        print("Temp 1 float 0", file=outfile)
        print("Density 1 float 0", file=outfile)
        print("B 4 float 0 0 0 1", file=outfile)
        print("V 3 float 0 0 0", file=outfile)
        print("Speed 1 float 0", file=outfile)
        
        # Write the points and point attributes
        # For each magnetic field line
        for i in range(self.nfdl):
            # Get the number of points on the line
            nk = self.npoints[i]
            
            # Get the coordinates of the i-th field line
            phi   = self.fdlphi[i][0:nk]
            theta = self.fdlth[i][0:nk]
            r     = self.fdlr[i][0:nk]
            
            # Get the attribute data
            temp  = self.fdltemp[i][0:nk]
            
            bphi   = self.fdlBphi[i][0:nk]
            btheta = self.fdlBth[i][0:nk]
            br     = self.fdlBr[i][0:nk]
            bmag   = numpy.sqrt( br*br + btheta*btheta + bphi*bphi )
            
            vphi   = self.fdlVphi[i][0:nk]
            vtheta = self.fdlVth[i][0:nk]
            vr     = self.fdlVr[i][0:nk]
            speed = numpy.sqrt( vr*vr + vtheta*vtheta + vphi*vphi )
        
            density = self.fdldensity[i][0:nk]

            # If we want the field lines to be on the rounded surface of the sun
            # (This seems like the opposite of cartesian to me? Keeping this name for consistency with other functions)
            if cartesian:
                xyz = ptr2xyz( phi, theta, r )
                for k in range(nk):
                    print("%g %g %g 1 (%g %g %g %g %g %g %g %g %g %g) " % (xyz[k,0], xyz[k,1], xyz[k,2], temp[k], density[k], bphi[k],btheta[k],br[k],bmag[k], vphi[k],vtheta[k],vr[k], speed[k]), file=outfile)
            else:
                for k in range(nk):
                    print("%g %g %g 1 (%g %g %g %g %g %g %g %g %g %g) " % (  phi[k], theta[k], r[k], temp[k], density[k], bphi[k],btheta[k],br[k],bmag[k], vphi[k],vtheta[k],vr[k], speed[k] )  , file=outfile)
         
           
        # Primitive attribute definition
        # These attributes are for each field line
        print("PrimitiveAttrib", file=outfile)
        print("Color 1 float 0", file=outfile)
        
        # Write the primitives and primitive attributes
        # Each line is its own primitive - specifically, it is an open polygon
        currentTotalOfPoints = 0
        print("Run " + str(self.nfdl) + " Poly" , file=outfile)

        for i in range(self.nfdl):
            nk = self.npoints[i]
            numVertices = str(nk)

            polyType = " : " # "<" is for closed, ":" is for open
            print(numVertices + polyType, end='', file=outfile)
            for k in range(nk):
                vertexIndex = currentTotalOfPoints + k
                print("%g " % vertexIndex, end='', file=outfile)
            currentTotalOfPoints += nk
            print("[%g]" % (self.color[i]), file=outfile)
        
        # Detail attribute definition
        # These attributes are for the entire file
        print("Attrib", file=outfile)
        print("Time 1 float 0", file=outfile)
        print("NR 1 int 0", file=outfile)
        print("NTh 1 int 0", file=outfile)
        print("NPh 1 int 0", file=outfile)
        print("NFdl 1 int 0", file=outfile)
        print("MaxPts 1 int 0", file=outfile)
        print("RSol 1 float 0", file=outfile)
        
        # Write the detail attributes
        print("(%g %d %d %d %d %d %g) " % (self.time, self.nr, self.nth, self.nph, self.nfdl, self.maxpts, self.rsol), file=outfile)
            
        # EOF
        print("beginExtra", file=outfile)
        print("endExtra", file=outfile)

    def defineBgeoAttrib(self, outfile, name, valueType, size, defaultValue):
        """Declares an attribute in a .bgeo file."""
        typeInt = 0

        if valueType == 'f':
            typeInt = 0  # float32
        elif valueType == 'i':
            typeInt = 1  # int32
        elif valueType == 'x':
            typeInt = 4  # index
        elif valueType == 'f':
            typeInt = 5  # vector of float32s
        else:
            raise Exception("Invalid Attribute Type. Please use 0 (float), 1 (int), 4 (index), or 5 (vector)")
        
        outfile.write(struct.pack('>' + 'h', len(name))) # length of name 
        outfile.write(bytearray(name)) # name

        # Index type has its own different rules
        if typeInt == 4:
            outfile.write(struct.pack('>' + 'h', 1))
            outfile.write(struct.pack('>' + 'I', typeInt))
            outfile.write(struct.pack('>' + 'i', size))
            for v in defaultValue:
                outfile.write(struct.pack('>' + 'h', len(v))) # length of default value
                outfile.write(bytearray(list(v)))

        # For all other types
        else:
            outfile.write(struct.pack('>' + 'h', size)) # size
            outfile.write(struct.pack('>' + 'I', typeInt)) # type
            if(size > 1):
                for v in defaultValue:
                    outfile.write(struct.pack('>' + valueType, v)) # default value
            else:
                outfile.write(struct.pack('>' + valueType, defaultValue)) # default value

    def writebgeo(self, outf, cartesian):
        """Takes the yfan data and writes a .BGEO file to be read by Houdini"""

        USHRT_MAX = 65535  # The largest number that a uint16 can be
        
        outfile = open(outf, 'wb')
        
        # Count the total number of points on all of the field lines.
        npoints = 0
        for i in range(self.nfdl):
            nk = self.npoints[i]
            npoints += nk
            
        # Write.BGEO header
        header = ['B', 'g', 'e', 'o', 'V']
        outfile.write(bytearray(header))
        outfile.write(struct.pack('>' + 'i', 5)) # version
        npointattrib = 6 if (self.Vr is not None) else 3
        nprimattrib = 2 if (self.lineids is not None) else 1
        nattrib = 8
        counts =  [npoints, self.nfdl, 0, 0, npointattrib, 0, nprimattrib, nattrib] # npoints, nprims, npointgroups, nprimgroups, npointattrib, nvertexattrib, nprimattrib, nattrib
        outfile.write(struct.pack('>'+'i'*len(counts), *counts))

        # Point attribute definition
        # These attributes are for each point
        self.defineBgeoAttrib(outfile, "B", 'f', 3, [0,0,0])
        self.defineBgeoAttrib(outfile, "BMag", 'f', 1, 0)
        self.defineBgeoAttrib(outfile, "Temp", 'f', 1, 0)
        if (self.Vr is not None):
                    self.defineBgeoAttrib(outfile, "Density", 'f', 1, 0)
                    self.defineBgeoAttrib(outfile, "V", 'f', 3, [0,0,0])
                    self.defineBgeoAttrib(outfile, "Speed", 'f', 1, 0)
        
        # Write the points and point attributes
        # For each magnetic field line
        for i in range(self.nfdl):
            # Get the number of points on the line
            nk = self.npoints[i]
            
            # Get the coordinates of the i-th field line
            phi   = self.fdlphi[i][0:nk]
            theta = self.fdlth[i][0:nk]
            r     = self.fdlr[i][0:nk]
            
            # Get the attribute data
            temp  = self.fdltemp[i][0:nk]
            
            bphi   = self.fdlBphi[i][0:nk]
            btheta = self.fdlBth[i][0:nk]
            br     = self.fdlBr[i][0:nk]
            bmag   = numpy.sqrt( br*br + btheta*btheta + bphi*bphi )
            
        
            if (self.Vr is not None):
                vphi   = self.fdlVphi[i][0:nk]
                vtheta = self.fdlVth[i][0:nk]
                vr     = self.fdlVr[i][0:nk]
                speed = numpy.sqrt( vr*vr + vtheta*vtheta + vphi*vphi )
        
                density = self.fdldensity[i][0:nk]

            # If we want the field lines to be on the rounded surface of the sun
            if cartesian:
                xyz = ptr2xyz( phi, theta, r )
                for k in range(nk):
                    if (self.Vr is not None):
                        points = [ xyz[k,0], xyz[k,1], xyz[k,2], 1, bphi[k],btheta[k],br[k],bmag[k], temp[k], density[k], vphi[k],vtheta[k],vr[k], speed[k] ]
                    else:
                        points = [xyz[k,0], xyz[k,1], xyz[k,2], 1, bphi[k],btheta[k],br[k],bmag[k], temp[k]]
                    outfile.write(struct.pack('>'+'f'*len(points), *points))

            else:
                for k in range(nk):
                    if (self.Vr is not None):
                        points = [phi[k], theta[k], r[k], 1, bphi[k],btheta[k],br[k],bmag[k], temp[k], density[k], vphi[k],vtheta[k],vr[k], speed[k]]
                    else:
                        points = [phi[k], theta[k], r[k], 1, bphi[k],btheta[k],br[k],bmag[k], temp[k]]
                    outfile.write(struct.pack('>'+'f'*len(points), *points))
         
        # Primitive attribute definition
        # These attributes are for each field line
        self.defineBgeoAttrib(outfile, "Color", 'f', 1, 0)
        if (self.lineids is not None):
            self.defineBgeoAttrib(outfile, "LineID", 'i', 1, 0)
        
        # Write the primitives and primitive attributes
        # Each line is its own primitive - specifically, it is an open polygon
        currentTotalOfPoints = 0
        outfile.write(struct.pack('>' + 'I', 0xffffffff))  # Run
        outfile.write(struct.pack('>' + 'H', int(self.nfdl)))  # Run length
        outfile.write(struct.pack('>' + 'i', 0x00000001))  # Polygon
        
        for i in range(self.nfdl):
            nk = self.npoints[i]

            polyType = ":" # "<" is for closed, ":" is for open
            outfile.write(struct.pack('>' + 'i', nk))
            outfile.write(struct.pack('>' + 'c', polyType))
            for k in range(nk):
                vertexIndex = currentTotalOfPoints + k
                if npoints > USHRT_MAX:
                    outfile.write(struct.pack('>' + 'i', int(vertexIndex)))
                else:
                    outfile.write(struct.pack('>' + 'H', int(vertexIndex)))
            currentTotalOfPoints += nk
            outfile.write(struct.pack('>' + 'f', self.color[i]))   
            if (self.lineids is not None):
                outfile.write(struct.pack('>' + 'i', self.lineids[i]))
        
        # Detail attribute definition
        # These attributes are for the entire file       
        self.defineBgeoAttrib(outfile, "Time", 'f', 1, 0)
        self.defineBgeoAttrib(outfile, "RSol", 'f', 1, 0)
        self.defineBgeoAttrib(outfile, "NR", 'i', 1, 0)
        self.defineBgeoAttrib(outfile, "NTh", 'i', 1, 0)
        self.defineBgeoAttrib(outfile, "NPh", 'i', 1, 0)
        self.defineBgeoAttrib(outfile, "NFdl", 'i', 1, 0)
        self.defineBgeoAttrib(outfile, "MaxPts", 'i', 1, 0)

        # Varmaps definition (detail attributes for local variable mapping)
        totalAttrib = npointattrib + nprimattrib + nattrib - 1
        varmaps = ["Temp -> Temp", "B -> B", "BMag -> BMag",  # Point attributes
                   "Color -> Color",  # Primitive attribute
                   "Time -> Time", "RSol -> Rsol", "NR -> NR", "NTh -> NTh", "NPh -> NPh", "NFdl -> NFdl", "MaxPts -> MaxPts"]  # Detail attributes
        if (self.Vr is not None):
            varmaps.extend(["V -> V", "Speed -> Speed", "Density -> Density"])
        if (self.lineids is not None):
            varmaps.extend(["LineID -> LineID"])
        self.defineBgeoAttrib(outfile, "varmap", 'x', totalAttrib, varmaps)
        
        # Write the detail attributes
        intAttribs = [self.nr, self.nth, self.nph, self.nfdl, self.maxpts]
        floatAttribs = [self.time, self.rsol]
        outfile.write(struct.pack('>'+'f'*len(floatAttribs), *floatAttribs))
        outfile.write(struct.pack('>'+'i'*len(intAttribs), *intAttribs))
        outfile.write(struct.pack('>'+'i', 0))
            
        # EOF
        outfile.write(chr(0x00)) #beginExtra
        outfile.write(chr(0xff)) #endExtra

    def current( self, radialscale ):
        # magnitude of curl(B) = current
        # That wouldn't be hard, but let's not bother now.
        # The effect of not doing it: we exaggerate the current in the areas where cells are large. - slevy
        Br_ph, Br_th, Br_r = numpy.gradient( self.Br )
        Bth_ph, Bth_th, Bth_r = numpy.gradient( self.Bth )
        Bph_ph, Bph_th, Bph_r = numpy.gradient( self.Bphi )

        # To compute partial derivatives, we divide ...
        #  Br_ph, Bth_ph, Bph_ph by the cell spacing along the phi direction,
        #  Br_th, Bth_th, Bph_th by the cell spacing along the theta direction, and
        #  Br_r, Bth_r, Bph_r by the cell spacing along the r direction.

        # expand reciprocal cell spacings into arrays we can multiply (phi,theta,r) grids by
        if self.oldcurrent:
            inv_ph = inv_th = inv_r = 1.0
        else:
            inv_ph = ( 1.0 / numpy.gradient( self.phCoord ) ) [ :, numpy.newaxis, numpy.newaxis ]

            inv_th = ( 1.0 / numpy.gradient( self.thCoord ) ) [ numpy.newaxis, :, numpy.newaxis ]

            inv_r =  ( 1.0 / numpy.gradient( self.rCoord ) ) [ numpy.newaxis, numpy.newaxis, : ]

        # numpy array broadcasting does what we want for radial-scaling,
        # since 'r' is the last (fastest) axis for this array.
        # otherwise we'd use a [ numpy.newaxis, ... ] trick for it too.

        # cross: y_z-z_y, z_x-x_z, x_y-y_x
        return numpy.sqrt( \
                (Bth_r*inv_r   - Br_th*inv_th)**2 + \
                (Br_ph*inv_ph  - Bph_r*inv_r)**2 + \
                (Bph_th*inv_th - Bth_ph*inv_ph)**2 ) \
             * radialscale

    # Compute an array of scale factors, indexed by radial position, for normalizing functions that vary a lot with radial position
    # To start with, radialnorm is an array of 3 numbers [ A, B, C ]
    # used as:    norm( r ) = exp( A + B*(r-1.0) + C*(r-1.0)**2 ]
    # Returns a numpy array of length self.nr.
    # If radialnorm=None, just returns an array of all ones.
    def radialfunc( self, radialnorm ):
        if radialnorm is None:
            return numpy.ones( (self.nr,) )

        elif len(radialnorm) == 3:
            return numpy.exp( radialnorm[0] + radialnorm[1]*(self.rCoord - 1.0) + radialnorm[2]*(self.rCoord - 1.0)**2 )

        else:
            raise Exception("Invalid radialnorm. Please use either None or an array of 3 numbers [A, B, C]")
        

    def writecurrentspeck( self, outf, cartesian = False, every = 1.0, currentthresh = 1.0, radialnorm = None, currentmin = 0 ):
        if isinstance(outf, str):
            outf = open(outf, 'w')

        print("#! /usr/bin/env partiview\n", file=outf)

        print("## ", " ".join(sys.argv), file=outf)

        # use radialnorm[] parameters to make a radial scaling function (array)
        rscaling = self.radialfunc( radialnorm )

        # for testing, report the range of radial scale factors as a comment in the speck file - innermost, mid, outermost
        rmid = int(self.nr/2)
        print("## radialnorm: %s => r[0]%g => %g;   r[%d]%g => %g;   r[%d]%g => %g" % \
              ( str(radialnorm),			\
                self.rCoord[0], rscaling[0],		\
                rmid, self.rCoord[rmid], rscaling[rmid], \
                self.nr-1, self.rCoord[-1], rscaling[-1] ), \
            file=outf)

        print("maxcomment 0", file=outf)
        print("datavar 0 current", file=outf)

        Jmag = self.current( 1.0 / rscaling )   # magnitude of current, with radial normalization
        ph = numpy.empty( (self.nr, ) )
        th = numpy.empty( (self.nr, ) )
        for i in range(self.nph):
            ph[:] = self.phCoord[i]
            for j in range(self.nth):
                th[:] = self.thCoord[j]
                xyzs = ptr2xyz( ph, th, self.rCoord )
                for k in range(self.nr):
                    if Jmag[i,j,k] > currentmin and (every<=1 or Jmag[i,j,k] > currentthresh or random.random() * every < 1):
                        if cartesian:
                            print("%g %g %g %g" % (xyzs[k,0], xyzs[k,1], xyzs[k,2], Jmag[i,j,k]), file=outf)
                        else:
                            print("%g %g %g %g" % ( self.phCoord[i], self.thCoord[j], self.rCoord[k], Jmag[i,j,k] ), file=outf)


    def getBoundingBox(self, currPoint, currMin, currMax):
        """Compares the current point against the current min and max, to get the bounding box"""
        if currPoint[0] < currMin[0]:
            currMin[0] = currPoint[0]
        if currPoint[1] < currMin[1]:
            currMin[1] = currPoint[1]
        if currPoint[2] < currMin[2]:
            currMin[2] = currPoint[2]

        if currPoint[0] > currMax[0]:
            currMax[0] = currPoint[0]
        if currPoint[1] > currMax[1]:
            currMax[1] = currPoint[1]
        if currPoint[2] > currMax[2]:
            currMax[2] = currPoint[2]

        return currMin, currMax


    def writevolumebgeo(self, outf, cartesian = False, every = 1.0, currentthresh = 0.0, radialnorm = None, randomize = False):
        """Writes out a volume .bgeo to be read by Houdini. This includes the current."""

        import scipy.spatial

        # Initialize constants
        bgeoVersion = 5
        npoints = self.nr * self.nth * self.nph
        nprims = 0
        npointgroups = 0
        nprimgroups = 0
        npointattrib = 7  # Do we also want to add Bmag and Speed? The field line data has these. Though, these are point attributes, so that may significantly increase the size of the file.
        nvertexattrib = 0
        nprimattrib = 0
        nattrib = 16
        totalAttrib = npointattrib + nvertexattrib + nprimattrib + nattrib - 1  # subtract 1 because "varmap" isn't an attribute in the traditional sense
        varmaps = ["Temp -> Temp", "Density -> Density", "B -> B", "Current -> Current", "V -> V", "NearestNeighbor -> NearestNeighbor", "FarthestNeighbor -> FarthestNeighbor",
                   "Time -> Time", "RSol -> Rsol", "NR -> NR", "NTh -> NTh", "NPh -> NPh", "NFdl -> NFdl", "MaxPts -> MaxPts", 
                   "VoxelSize -> VoxelSize", "VoxelSizeMax -> VoxelSizeMax", "MinX -> MinX", "MinY -> MinY", "MinZ -> MinZ", "MaxX -> MaxX", "MaxY -> MaxY", "MaxZ -> MaxZ"]

        # Open file
        if isinstance(outf, str):
            outf = open(outf, 'wb')

        # Borders of the bounding box 
        bbMin = [numpy.Infinity, numpy.Infinity, numpy.Infinity]
        bbMax = [-numpy.Infinity, -numpy.Infinity, -numpy.Infinity]

        # Calculate current
        rscaling = self.radialfunc( radialnorm )  # use radialnorm[] parameters to make a radial scaling function (array)
        Jmag = self.current( 1.0 / rscaling )  # magnitude of current, with radial normalization

        # Save Points
        data = []
        ph = numpy.empty( (self.nr,) )
        th = numpy.empty( (self.nr,) )
        numpy.random.seed(42)
        for iph in range(self.nph):
            ph[:] = self.phCoord[iph]
            for ith in range(self.nth):
                th[:] = self.thCoord[ith]
                xyzs = ptr2xyz(ph, th, self.rCoord)
                for ir in range(self.nr):
                    if ((not randomize) and (iph%every==0 and ith%every==0 and ir%every==0)) or (randomize and (every<=1 or numpy.random.random() * every*every*every < 1)):
                            coord = numpy.array((numpy.NAN, numpy.NAN, numpy.NAN))
                            if cartesian:
                                coord = numpy.array((xyzs[ir,0], xyzs[ir,1], xyzs[ir,2]))
                            else:
                                coord = numpy.array((self.phCoord[iph], self.thCoord[ith], self.rCoord[ir]))

                            current = Jmag[iph,ith,ir]
                            if current <= currentthresh:
                                current = 0

                            bbMin, bbMax = self.getBoundingBox(coord, bbMin, bbMax)

                            point = [coord[0], coord[1], coord[2], 1.0,
                                        self.temperature[iph][ith][ir],
                                        self.density[iph][ith][ir],
                                        self.Bphi[iph][ith][ir], self.Bth[iph][ith][ir], self.Br[iph][ith][ir],
                                        current,
                                        self.Vphi[iph][ith][ir], self.Vth[iph][ith][ir], self.Vr[iph][ith][ir]]
                            data.append(point)

        data = numpy.array(data)

        # Get point distances
        positions = data[:,:3]
        bucket = 8  # stop subdividing KDTree nodes when there are this few points in them
        tree = scipy.spatial.cKDTree(positions, bucket)
        nnearest = 6+1  # how many nearest neighbors to return
        approx = 0.1
        dists, indices = tree.query(positions, k=nnearest, eps=approx)

        # Write Header
        header = ['B', 'g', 'e', 'o', 'V']
        outf.write(bytearray(header))
        counts =  [bgeoVersion, len(data), nprims, npointgroups, nprimgroups, npointattrib, nvertexattrib, nprimattrib, nattrib]
        outf.write(struct.pack('>'+'i'*len(counts), *counts))
        
        # Define Point Attributes
        self.defineBgeoAttrib(outf, "Temp", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "Density", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "B", 'f', 3, [0,0,0])
        self.defineBgeoAttrib(outf, "Current", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "V", 'f', 3, [0,0,0])
        self.defineBgeoAttrib(outf, "NearestNeighbor", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "FarthestNeighbor", 'f', 1, 0)

        # Write points
        distanceMin = numpy.Infinity
        distanceMax = -numpy.Infinity
        for i in range(len(data)):
            point = data[i]
            for num in point:
                outf.write(struct.pack('>' + 'f', float(num)))
            # Compare/write point distances
            nearestNeighbor = dists[i][1]
            farthestNeighbor = dists[i][3]
            if nearestNeighbor < distanceMin:
                distanceMin = nearestNeighbor
            if farthestNeighbor > distanceMax:
                distanceMax = farthestNeighbor
            outf.write(struct.pack('>' + 'f', nearestNeighbor))
            outf.write(struct.pack('>' + 'f', farthestNeighbor))

        # Define Details Attributes
        self.defineBgeoAttrib(outf, "NR", 'i', 1, 0)
        self.defineBgeoAttrib(outf, "NTh", 'i', 1, 0)
        self.defineBgeoAttrib(outf, "NPh", 'i', 1, 0)
        self.defineBgeoAttrib(outf, "NFdl", 'i', 1, 0)
        self.defineBgeoAttrib(outf, "MaxPts", 'i', 1, 0)
        self.defineBgeoAttrib(outf, "Time", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "RSol", 'f', 1, 0)

        self.defineBgeoAttrib(outf, "VoxelSize", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "VoxelSizeMax", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "MinX", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "MinY", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "MinZ", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "MaxX", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "MaxY", 'f', 1, 0)
        self.defineBgeoAttrib(outf, "MaxZ", 'f', 1, 0)

        self.defineBgeoAttrib(outf, "varmap", 'x', totalAttrib, varmaps)
        
        intAttribs = [self.nr, self.nth, self.nph, self.nfdl, self.maxpts]
        floatAttribs = [self.time, self.rsol, distanceMin/2, distanceMax/2, bbMin[0], bbMin[1], bbMin[2], bbMax[0], bbMax[1], bbMax[2]]
        outf.write(struct.pack('>'+'i'*len(intAttribs), *intAttribs))
        outf.write(struct.pack('>'+'f'*len(floatAttribs), *floatAttribs))
        outf.write(struct.pack('>'+'i', 0))  # For varmap

        # Mark end of file
        outf.write(chr(0x00))
        outf.write(chr(0xff))

        

    def writecurrentbgeo( self, outf, cartesian = False, every = 1.0, currentthresh = 1.0, radialnorm = None, subsample=1 ):
        """Writes out a .bgeo file of the current, as points, to be read by Houdini."""
        if isinstance(outf, str):
            outf = open(outf, 'wb')

        # use radialnorm[] parameters to make a radial scaling function (array)
        rscaling = self.radialfunc( radialnorm )
        Jmag = self.current( 1.0 / rscaling )  # magnitude of current, with radial normalization

        data = []

        ph = numpy.empty( (self.nr, ) )
        th = numpy.empty( (self.nr, ) )
        for i in range(self.nph):
            ph[:] = self.phCoord[i]
            for j in range(self.nth):
                th[:] = self.thCoord[j]
                xyzs = ptr2xyz( ph, th, self.rCoord )
                for k in range(self.nr):
                    if every<=1 or Jmag[i,j,k] > currentthresh or random.random() * every < 1: 
                        if cartesian:
                            data.append(xyzs[k,0])
                            data.append(xyzs[k,1])
                            data.append(xyzs[k,2])
                            data.append(1.0)
                            data.append(Jmag[i,j,k])
                        else:
                            data.append(self.phCoord[i])
                            data.append(self.thCoord[j])
                            data.append(self.rCoord[k])
                            data.append(1.0)
                            data.append(Jmag[i,j,k])

        # Write.BGEO header
        header = ['B', 'g', 'e', 'o', 'V']
        outf.write(bytearray(header))
        outf.write(struct.pack('>' + 'i', 5)) # version
        counts =  [len(data)/5, 0, 0, 0, 1, 0, 0, 0] # npoints, nprims, npointgroups, nprimgroups, npointattrib, nvertexattrib, nprimattrib, nattrib
        outf.write(struct.pack('>'+'i'*len(counts), *counts))

        # Point attribute definition
        self.defineBgeoAttrib(outf, "Current", 'f', 1, 0.0)

        # Write points
        for datum in data:
            outf.write(struct.pack('>' + 'f', float(datum)))

        # EOF
        outf.write(chr(0x00)) #beginExtra
        outf.write(chr(0xff)) #endExtra


    def radial_profile( self, field ):
        # indexed by phi, theta, r
        rp = []
        for ir in range(self.nr):
            vmin = field[:,:,ir].min()
            vmax = field[:,:,ir].max()
            vmean = field[:,:,ir].mean()
            rp.append( ( vmin, vmax, vmean ) )
        return numpy.array( rp )
        

    def print_stats(self, outf = sys.stdout):
        dr = self.rCoord[1:] - self.rCoord[0:-1]
        dth = self.thCoord[1:] - self.thCoord[0:-1]
        dph = self.phCoord[1:] - self.phCoord[0:-1]

        rmin,  rmax =  self.rCoord[0],  self.rCoord[-1]
        thmin, thmax = self.thCoord[0], self.thCoord[-1]
        phmin, phmax = self.phCoord[0], self.phCoord[-1]

        drmin, drmax = dr.min(), dr.max()
        rdthmin, rdthmax = rmin*dth.min(), rmax*dth.max()
        rdphmin, rdphmax = rmin*dph.min(), rmax*dph.max()


        print("cell sizes (dr,r*dtheta,r*phi)", file=outf)
        print("  min:      %10g %10g %10g" % ( drmin, rdthmin, rdphmin ), file=outf)
        print("  max:      %10g %10g %10g" % ( drmax, rdthmax, rdphmax ), file=outf)
        print("  max/min:  %10.3g %10.3g %10.3g" % ( drmax/drmin, rdthmax/rdthmin, rdphmax/rdphmin ), file=outf)
        print("  span/min: %10.4g %10.4g %10.4g" % ( (rmax-rmin)/drmin,  (thmax-thmin)*rmax/rdthmin, (phmax-phmin)*rmax/rdphmin ), file=outf)

        Bmag = numpy.sqrt(self.Br**2 + self.Bth**2 + self.Bphi**2)
        print("\nB field         (|B|, Br, Btheta, Bphi)", file=outf)
        print("  min:  %10g  %10g %10g %10g" % ( Bmag.min(), self.Br.min(), self.Bth.min(), self.Bphi.min() ), file=outf)
        print("  max:  %10g  %10g %10g %10g" % ( Bmag.max(), self.Br.max(), self.Bth.max(), self.Bphi.max() ), file=outf)
        print(" mean:  %10g  %10g %10g %10g" % ( Bmag.mean(), self.Br.mean(), self.Bth.mean(), self.Bphi.mean() ), file=outf)

        if self.Vr is None:
            print("\nVelocity field not loaded", file=outf)
        else:
            Vmag = numpy.sqrt(self.Vr**2 + self.Vth**2 + self.Vphi**2)
            print("\nVelocity field  (|V|, Vr, Vtheta, Vphi)", file=outf)
            print("  min:  %10g  %10g %10g %10g" % ( Vmag.min(), self.Vr.min(), self.Vth.min(), self.Vphi.min() ), file=outf)
            print("  max:  %10g  %10g %10g %10g" % ( Vmag.max(), self.Vr.max(), self.Vth.max(), self.Vphi.max() ), file=outf)
            print(" mean:  %10g  %10g %10g %10g" % ( Vmag.mean(), self.Vr.mean(), self.Vth.mean(), self.Vphi.mean() ), file=outf)

        if self.density is None:
            print("\nDensity, temperature not loaded", file=outf)
        else:
            print("\nDensity, temperature", file=outf)
            print("  min:  %10g  %10g" % ( self.density.min(), self.temperature.min() ), file=outf)
            print("  max:  %10g  %10g" % ( self.density.max(), self.temperature.max() ), file=outf)
            print(" mean:  %10g  %10g" % ( self.density.mean(), self.temperature.mean() ), file=outf)

    def print_rprofile( self, outf ):
        rD = (1/1e-20) * self.radial_profile( self.density )
        rT = (1/1e6) * self.radial_profile( self.temperature )

        Bmag = numpy.sqrt( self.Br**2 + self.Bth**2 + self.Bphi**2 )
        rBmag = self.radial_profile( Bmag )

        Jmag = self.current( 1.0 )   # raw unscaled current
        rJmag = self.radial_profile( Jmag )

        rvel = self.radial_profile( self.Vr )

        print("\nRadial Profiles (Density/1e-20   Temperature/1e6)", file=outf)
        print("r\tDensity\tmin  max   mean    Temp\tmin  max  mean   Bmag\tmin  max  mean  current\tmin  max  mean  Vr\tmin  mean  max", file=outf)
        for i in range(self.nr):
            print("%g\t%g %g %g\t" % (self.rCoord[i], rD[i,0], rD[i,1], rD[i,2]), \
                          "\t%g %g %g" % (rT[i,0], rT[i,1], rT[i,2]), \
                          "\t%g %g %g" % (rBmag[i,0], rBmag[i,1], rBmag[i,2]), \
                          "\t%g %g %g" % (rJmag[i,0], rJmag[i,1], rJmag[i,2]), \
                          "\t%g %g %g" % (rvel[i,0], rvel[i,1], rvel[i,2]), \
                file=outf)

ii = 1
output_type = 1  # 0 for traj, 1 for speck, 2 for geo, 3 for Jspeck, 4 for bgeo, 5 for bgeo volume
outfname = None
#cartesian = False
cartesian = True
verbose = False
rprofile = False
every = 1.0
currentthresh = 100.0
randomize = False

#currentnorm = [ 0, -2.0, 0.2 ]	# B normalization: exp( a + b*(r-1.0) + c*(r-1.0)**2 ).   This leaves about 20-fold radial falloff, instead of about 3000.   Maybe this is too much flattening.

currentnorm = None  # only normalize current if -Bnorm says to
oldcurrent = False
currentmin = 0

otypes = { "t":0, "s":1, "g":2, "j":3, "b":4, "d":5 }

while len(sys.argv) > ii+1 and sys.argv[ii][0] == "-":
    a = sys.argv[ii]
    if a == "-v":
        verbose = True
        ii += 1

    elif a == "-e":
        ee = [ float(s) for s in sys.argv[ii+1].split(',') ]
        every = ee[0]
        if len(ee) >= 2:
            currentthresh = ee[1]
        if len(ee) >=3:
            if ee[2] == 1:
                randomize = True
        ii += 2

    elif a == "-Bnorm":
        # -Bnorm  deltaA,deltaB[,deltaC][!]
        ss = sys.argv[ii+1]
        incremental = 1
        if ss[-1] == '!':
            ss = ss[:-2]
            incremental = 0
        cn = [ float(s) for s in sys.argv[ii+1].split(',') ]
        if currentnorm is None:
            currentnorm = cn
        else:
            for i in range(len(cn)):
                currentnorm[i] = cn[i] + currentnorm[i]*incremental
        ii += 2

    elif a == "-Bslices":
        cx,cy,cz,cr = [ float(s) for s in sys.argv[ii+1] ]
        ii += 2

    elif a == "-r":
        rprofile = True
        ii += 1

    elif a == "-c":
        cartesian = True
        ii += 1

    elif a == "-oldcurrent":
        oldcurrent = True
        ii += 1

    elif a == "-currentmin":
        currentmin = float(sys.argv[ii+1])
        ii += 2

    elif a[1] in otypes:	# -t, -s, -g, -j, -b, -d
        output_type = otypes[ a[1] ]
        outfname = sys.argv[ii+1]
        ii += 2
    else:
        print("Unknown option \"%s\"" % a, file=sys.stderr)
        print("Usage: %s [-v] [-c] [-e everyN] [-t outfile.traj | -s outfile.speck | -g outfile.geo | -b outfile.bgeo | -j current.speck | -d volume.bgeo]  yfanfile.dat" % sys.argv[0], file=sys.stderr)
        sys.exit(1)

if outfname is None and not (verbose or rprofile):
    print("Usage: %s [-v] [-c] [-e everyN[,currentthresh,randomize]] [-t outfile.traj | -s outfile.speck | -g outfile.geo | -b outfile.bgeo | -j current.speck | -d volume.bgeo]  yfanfile.dat" % sys.argv[0], file=sys.stderr)
    sys.exit(1)


yf = YFan(sys.argv[ii])

yf.oldcurrent = oldcurrent  # sigh.

if verbose:
    # Compute statistics
    # Peek into coord vars
    yf.print_stats( sys.stdout )

if rprofile:
    yf.print_rprofile( sys.stdout )



if outfname is not None:
    if output_type == 0:
        yf.writetraj( outfname, cartesian )
    elif output_type == 1:
        yf.writespeck( outfname, cartesian, currentnorm )
    elif output_type == 2:
        yf.writegeo( outfname, cartesian )
    elif output_type == 3:
        yf.writecurrentspeck( outfname, cartesian, every, currentthresh, currentnorm, currentmin )
    elif output_type == 4:
        yf.writebgeo( outfname, cartesian )
    elif output_type == 5:
        yf.writevolumebgeo( outfname, cartesian, every, currentthresh, currentnorm, randomize )
    elif output_type == 6:
        yf.writebspeck( outfname, cartesian, [cx,cy,cz,cr] )
