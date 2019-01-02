==========
Data model
==========

We model a contact matrix using three tables.

Tables
======

chroms
------

+ Required columns: ``name[, length]``
+ Order: *enumeration*

An semantic ordering of the chromosomes, scaffolds or contigs of the assembly that the data is mapped to. This information can be extracted from the bin table below, but is included separately for convenience. This enumeration is the intended ordering of the chromosomes as they would appear in a global contact matrix. Additional columns can provide metadata on the chromosomes, such as their length.

bins
----

+ Required columns: ``chrom, start, end [, weight]``
+ Order: ``chrom`` (*enum*), ``start``

An enumeration of the concatenated genomic bins that make up a single dimension or axis of the global contact matrix. Genomic bins can be of fixed size or variable sizes (e.g. restriction fragments). A genomic bin is defined by the triple (chrom, start, end), where start is zero-based and end is 1-based. The order is significant: the bins are sorted by chromosome (based on the chromosome enumeration) then by start, and each genomic bin is implicitly endowed with a 0-based bin ID from this ordering (i.e., its row number in the table). A reserved but optional column called ``weight`` can store weights for normalization or matrix balancing. Additional columns can be added to describe other bin-associated properties such as additional normalization vectors or bin-level masks.

pixels
------

+ Required columns: ``bin1_id, bin2_id, count``
+ Order: ``bin1_id, bin2_id``

The contact matrix is stored as a single table containing only the nonzero upper triangle elements, assuming the ordering of the bins given by the bin table. Each row defines a non-zero element of the contact matrix. Additional columns can be appended to store pixel-associated properties such as pixel-level masks or filtered and transformed versions of the data. Currently, the pixels are sorted lexicographically by the bin ID of the 1st axis (matrix row) then the bin ID of the 2nd axis (matrix column).


Sparse Array Representation
===========================


Text-based formats
==================

BG2
---


COO
---


Annotation
----------


.. comment:

	Why model it this way?

	To balance the tradeoff between simplicity, terseness and flexibility in an attempt to stay `Zen <https://www.python.org/dev/peps/pep-0020/>`_. 

	+ The schema is flexible enough to describe a whole genome contact matrix, or any subset of a contact matrix, including single contig-contig tiles.
	+ Given the variety of ways we might want to read the data or add new columns, flatter is better than nested.
	+ For one, it makes the data much easier to stream and process in chunks, which ideal for many types of out-of-core algorithms on very large contact matrices.
	+ Separating bins (annotations of the axis labels) from pixels (the matrix data) allows for easy inclusion of bin-level properties without introducing redundancy.


	Note that this flat structure [combination of bin + pixel tables] also defines a companion plain text format, a simple serialization of the binary format. Two forms are possible:

	- Two-file: The bin table and pixel table are stored as separate tab-delimited files (BED file + sparse triple file). See the output format from `Hi-C Pro <http://nservant.github.io/HiC-Pro/RESULTS.html#intra-and-inter-chromosomal-contact-maps>`_.

	- Single-file ("merged"): The ``bin1_id`` and ``bin2_id`` columns of the pixel table are replaced with annotations from the bin table, suffixed with `1` or `2` accordingly (e.g. ``chrom1``, ``start1``, ``end1``, ``weight1``, etc.). The result is a 2D extension of the `bedGraph <https://genome.ucsc.edu/goldenpath/help/bedgraph.html>`_ track format.


.. comment:
	Notes
	~~~~~

	Column-oriented vs record-oriented tables
	^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	Why is the reference schema column-oriented?

	- Cheap column addition/removal.
	- Better compression ratios.
	- Blazingly fast I/O speed can be achieved with new compressors such as `blosc <http://www.blosc.org/>`_.
	- Easy to migrate to other column stores such as `bcolz <https://github.com/Blosc/bcolz>`_, Apache `Parquet <https://parquet.apache.org/>`_, and Apache `Arrow <http://blog.cloudera.com/blog/2016/02/introducing-apache-arrow-a-fast-interoperable-in-memory-columnar-data-structure-standard/>`_.

	There is a tradeoff between flexibility and number of read cycles required to fetch all columns of a table, however, a column-oriented schema is fully interchangeable with a record-oriented representation (e.g., traditional SQL databases, CSV files).


	Supporting a matrix “view”
	^^^^^^^^^^^^^^^^^^^^^^^^^^

	Indexes are stored as 1D datasets in a separate group. The current indexes can be thought of as run-length encodings of the ``bins/chrom`` and ``pixels/bin1_id`` columns, respectively.


	Limitations
	^^^^^^^^^^^

	A complete rectangular matrix “view” of the data must be modeled on top of this representation. 2D range queries must be computed with the help of indexes. The sort order on the pixels and types of indexing strategies that can be used are strongly related. This could be changed in future versions of the schema.


.. comment:

    genome-assembly : string
        Name of genome assembly;  default: "unknown".

    Good h5py examples:
    https://www.uetke.com/blog/python/how-to-use-hdf5-files-in-python/

.. comment:
  Implementation Notes
  ====================

  Having the ``bin1_offset`` index, the ``bin1_id`` column becomes redundant, but we keep it for convenience as it is extremely compressible. It may be dropped in future versions.
