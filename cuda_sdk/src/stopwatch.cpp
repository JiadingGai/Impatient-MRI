/*
* Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and 
* proprietary rights in and to this software and related documentation and 
* any modifications thereto.  Any use, reproduction, disclosure, or distribution 
* of this software and related documentation without an express license 
* agreement from NVIDIA Corporation is strictly prohibited.
* 
*/

/* CUda UTility Library */

#include <vector>

// includes, file
#include <stopwatch.h>

// includes, project
#include <exception.h>

////////////////////////////////////////////////////////////////////////////////
// static variables

//! global index for all stop watches
#ifdef _WIN32
/*static*/ std::vector< StopWatchC* > StopWatchC::swatches;
#else
template<class OSPolicy>
/*static*/ std::vector< StopWatchBase<OSPolicy>* > 
StopWatchBase<OSPolicy>::    swatches;
#endif


// namespace, unnamed
namespace 
{
    // convenience typedef
    typedef  std::vector< StopWatchC* >::size_type  swatches_size_type;  

    //////////////////////////////////////////////////////////////////////////////
    //! Translate stop watch name to index
    //////////////////////////////////////////////////////////////////////////////
    swatches_size_type
    nameToIndex( const unsigned int& name) 
    {

#ifdef _DEBUG
        const swatches_size_type pos = name - 1;
        if(     (pos >= StopWatchC::swatches.size()) 
            || ( NULL == StopWatchC::swatches[pos])) 
        {
            RUNTIME_EXCEPTION( "No StopWatch with the requested name exist.");
        }

        return pos;
#else 
        return name - 1;
#endif
    }

} // end namespace, unnamed

// Stop watch
namespace StopWatch 
{
    //////////////////////////////////////////////////////////////////////////////
    //! Create a stop watch
    //////////////////////////////////////////////////////////////////////////////
    const unsigned int 
    create() 
    {
        // create new stopwatch
        StopWatchC* swatch = new StopWatchC();
        if( NULL == swatch) 
        {
            return 0;
        }

        // store new stop watch
        StopWatchC::swatches.push_back( swatch);

        // return the handle to the new stop watch
        return (unsigned int) StopWatchC::swatches.size();
    }

    //////////////////////////////////////////////////////////////////////////////
    // Get a handle to the stop watch with the name \a name
    //////////////////////////////////////////////////////////////////////////////
    StopWatchC& 
    get( const unsigned int& name) 
    {
        return *(StopWatchC::swatches[nameToIndex( name)]);
    }

    //////////////////////////////////////////////////////////////////////////////
    // Delete the stop watch with the name \a name
    //////////////////////////////////////////////////////////////////////////////
    void
    destroy( const unsigned int& name) 
    {
        // get index into global memory
        swatches_size_type  pos = nameToIndex( name);
        // delete stop watch
        delete StopWatchC::swatches[pos];
        // invalidate storage
        StopWatchC::swatches[pos] = NULL;
    }

} // end namespace, StopWatch
