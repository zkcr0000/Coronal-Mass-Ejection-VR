#! /usr/bin/env python3

from __future__ import print_function

import numpy

class PbReader(object):
    PBHMagic = 0xffffff98

    ## .pb file format:
    ##  int32  PBHMagic constant    (Can be used to determine the file's byte order)
    ##  int32  dataoff -- file offset of first point data record
    ##  int32  nfields -- number of attributes

    ##  then follows <nfields> attribute names, each followed by a \x00 byte
    ##  There may be additional data, e.g. comments, following the <nfields>'th attribute name

    ## Then, at file offset dataoff, is a sequence of point data records.
    ## Each point data record is 4*(1+3+nfields) bytes long:
    ##    int32  point-identifier.   This may be wasted space, but it's defined as part of the .pb format.
    ##    float32 * 3   X Y Z position
    ##    float32 * <nfields>  zero or more attributes for this point

    ## The format doesn't specify any sort of point count. The sequence of point data ends at the end of the file.

    def __init__(self, infname):
        self.infname = infname
        self.inf = open(infname, 'rb')

        # dataoff = 3*4 + flen + pad

        head = numpy.fromfile(self.inf, dtype=numpy.uint32, count=3)
        if head[0] != self.PBHMagic:
            raise ValueError("PbReader('%s'): doesn't look like a .pb file" % infname)

        dataoff = head[1]
        nfields = head[2]

        bfieldsetc = self.inf.read( dataoff - 3*4 )
        bfieldbits = bfieldsetc.split(b'\0', maxsplit=nfields)
        self.fieldnames = [b.decode() for b in bfieldbits[:-1]]
        if len(self.fieldnames) != nfields:
            raise ValueError("Expected %d fields in %s, got these %d instead: %s" % (nfields, infname, len(self.fieldnames), " ".join(self.fieldnames)))
        self.extradata = bfieldbits[-1]

        self.data = numpy.fromfile( self.inf, dtype=numpy.float32 ).reshape(-1, 4+nfields)  # almost right, except for particle id


    def lookup(self, fldname, insist=True):
        for ifield, fi in enumerate(self.fieldnames):
            if fi == fldname:
                return ifield
        if insist:
            raise ValueError("Couldn't find field %s in %s, only these: %s" % (fldname, self.infname, " ".join(self.fieldnames)))
        return None

    def getfields(self, fieldnames):
        n = len(fieldnames)
        ifields = [self.lookup(fldname, insist=True) for fldname in fieldnames]
        result = numpy.empty( (len(self.data), n), dtype=numpy.float32 )
        for k, ifield in enumerate(ifields):
            result[:,k] = self.data[:,ifield+4]
        return result


        
        

    
class PbWriter(object):
    """PbWriter writes collections of particles in .pb format.  Use as in:
    pb = PbWriter( outfname, [list_of_fieldnames] )
    followed by one or more calls to write particle data, any number of particles at a time
    pb.writepcles( xyz, fields, id )  # xyz is Nx3 numpy array, fields is Nxnfields, id is 1D integers (or None)
    pb.writepcles( morexyz, morefields, id )
    ...
    pb.close()
    """

    PBHMagic = 0xffffff98;

    def __init__(self, outf, fieldnames, pbextra=None):
        if isinstance(outf, (str,bytes)):
            self.outfname = outf
            self.outf = open(outf, 'wb')
        else:
            self.outfname = outf.name
            self.outf = outf

        self.fieldnames = fieldnames
        bfieldnames = [(fld.encode() if hasattr(fld, 'encode') else fld) for fld in fieldnames]
        self.pcount = 0

        flen = sum([len(bf) for bf in bfieldnames]) + len(bfieldnames)

        if pbextra is not None:
            bpbextra = b'\0PBEXTRA:\n' + (pbextra.encode() if hasattr(pbextra,'encode') else pbextra)
        else:
            bpbextra = b''

        flen += len(bpbextra)

        pad = ((flen + 3) & ~3) - flen
        dataoff = 3*4 + flen + pad

        head = numpy.array( [ self.PBHMagic, dataoff, len(bfieldnames) ], dtype=numpy.uint32 )
        head.tofile( self.outf )
        bfieldstr = b"\0".join(bfieldnames) + bpbextra + b"\0" * (pad+1)
        self.outf.write( bfieldstr )

    # now ready to append particles

    def writepcles(self, xyz, fields, id=None):
        if xyz.dtype != numpy.float32:
            xyz = xyz.astype( numpy.float32 )
        #if len(xyz) == 1:
        #    xyz = xyz.reshape(1,-1)
        #    fields = fields.reshape(1,-1)
        
        n = xyz.shape[0]

        if id is None:
            id = range(n)

        if not isinstance(id, numpy.ndarray):
            id = numpy.array( id, dtype=numpy.uint32 ).reshape( (n,) )

        if isinstance(fields, numpy.ndarray):
            totcolumns = fields.shape[1]
        else:
            totcolumns = 0
            for field in fields:
                if not isinstance(field, numpy.ndarray):
                    raise ValueError("PbWriter.writepcles(): fields must be numpy arrays")
                if field.shape[0] != n:
                    raise ValueError("PbWriter.writepcles(): fields must have as many rows as xyz point array but %d != %d" % (field.shape[0], n))
                if len(field.shape) == 1:
                    totcolumns += 1
                else:
                    totcolumns += field.shape[1]

        if totcolumns != len(self.fieldnames):
            raise ValueError("PbWriter.writepcles(): %d fieldnames but %d total columns worth of fields" % (len(self.fieldnames), totcolumns))

        pbtype = numpy.dtype( [('id', 'i4'), ('p', 'f4', (3,)), ('f', 'f4', (len(self.fieldnames), ) )] )

        pbdata = numpy.ndarray( (n,), dtype=pbtype )
        pbdata['id'] = id
        pbdata['p'] = xyz

        # arrange the fields
        if isinstance(fields, numpy.ndarray):
            pbdata['f'] = fields.astype(numpy.float32)
        else:
            # fields is a list/tuple of separate fields.  Dole them out
            o = 0
            for field in fields:
                if len(field.shape) == 1:
                    pbdata['f'][:, o] = field
                    o += 1
                else:
                    ncolumns = field.shape[1]
                    pbdata['f'][:, o:o+ncolumns] = field
                    o += ncolumns

        pbdata.tofile( self.outf )

    def close(self):
        if self.outf:
            self.outf.close()
        self.outf = None

    def __del__(self):
        self.close()

if __name__ == "__main__":
    import numpy

    # from pbio import PbWriter

    # Make some data - four variables defined on a sphere

    thetas = numpy.linspace( 0, 2*numpy.pi, num=120, endpoint=False )
    phis = numpy.linspace( -0.5*numpy.pi, 0.5*numpy.pi, num=60, endpoint=True )

    # spherical coordinates
    npts = len(thetas) * len(phis)
    xyz = numpy.zeros( shape=(npts, 3) )		  # particle positions
    ids = numpy.zeros( shape=(npts,), dtype=int )   # integer particle ids, even though we don't need them here
    attrs = numpy.zeros( shape=(npts, 4) )		  # particle attributes

    attrnames = ['theta', 'phi', 'simplefunc', 'wildfunc']


    # There are much better ways to fill in xyz[] and attrs[], but this is simple & clear
    i = 0
    for phi in phis:
        for theta in thetas:
            xyz[i,0] = numpy.cos(theta) * numpy.cos(phi)
            xyz[i,1] = numpy.sin(theta) * numpy.cos(phi)
            xyz[i,2] = numpy.sin(phi)
            attrs[i,0] = theta
            attrs[i,1] = phi
            attrs[i,2] = numpy.cos( 3 * theta )	# simple function
            attrs[i,3] = numpy.cos( 3 * theta + 3 * numpy.tan( 0.9*phi ) )  # wilder function
            ids[i] = i
            i += 1

    pbwriter = PbWriter( 'pbio-example.pb', attrnames )
    pbwriter.writepcles( xyz, attrs, id=ids )
    pbwriter.close()

    print("#! /usr/bin/env partiview")
    print("# Paste this into a script, e.g. pbio-example.cf, and run: partiview pbio-example.cf")
    print("pb pbio-example.pb")
    print("eval lum const 5")
    print("eval color wildfunc -1.01 1.01 # slightly bigger range than wildfunc's actual values")
    print("# \"color wildfunc\" or \"color simplefunc\" or \"color theta\" or \"color phi\"")
    print("eval jump -2.5 -5.3 3.5  51 -36 -38")

