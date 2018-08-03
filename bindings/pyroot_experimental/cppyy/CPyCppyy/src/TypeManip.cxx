// Bindings
#include "CPyCppyy.h"
#include "TypeManip.h"

// Standard
#include <ctype.h>


//- helpers ------------------------------------------------------------------
static inline
std::string::size_type find_qualifier_index(const std::string& name)
{
// Find the first location that is not part of the class name proper.
    std::string::size_type i = name.size() - 1;
    for ( ; 0 < i; --i) {
        std::string::value_type c = name[i];
        if (isalnum((int)c) || c == '>')
            break;
    }

    return i+1;
}

static inline void erase_const(std::string& name)
{
// Find and remove all occurrence of 'const'.
    std::string::size_type spos = std::string::npos;
    while ((spos = name.find("const") ) != std::string::npos) {
        std::string::size_type i = 5;
        while (name[spos+i] == ' ') ++i;
        name.swap(name.erase(spos, i));
    }
}

static inline void rstrip(std::string& name)
{
// Remove space from the right side of name.
    std::string::size_type i = name.size();
    for ( ; 0 < i; --i) {
       if (!isspace(name[i]))
           break;
    }

    if (i != name.size())
        name = name.substr(0, i);
}


//----------------------------------------------------------------------------
std::string CPyCppyy::TypeManip::remove_const(const std::string& cppname)
{
// Remove 'const' qualifiers from the given C++ name.
    std::string::size_type tmplt_start = cppname.find('<');
    std::string::size_type tmplt_stop  = cppname.rfind('>');
    if (tmplt_start != std::string::npos && tmplt_stop != std::string::npos) {
    // only replace const qualifying cppname, not in template parameters
        std::string pre = cppname.substr(0, tmplt_start);
        erase_const(pre);
        std::string post = cppname.substr(tmplt_stop+1, std::string::npos);
        erase_const(post);

       return pre + cppname.substr(tmplt_start, tmplt_stop+1) + post;
    }

    std::string clean_name = cppname;
    erase_const(clean_name);
    return clean_name;
}


//----------------------------------------------------------------------------
std::string CPyCppyy::TypeManip::clean_type(
    const std::string& cppname, bool template_strip, bool const_strip)
{
// Strip C++ name from all qualifiers and compounds.
    std::string::size_type i = find_qualifier_index(cppname);
    std::string name = cppname.substr(0, i);
    rstrip(name);

    if (name.back() == ']') {                      // array type?
    // TODO: this fails templates instantiated on arrays (not common)
        name = name.substr(0, name.find('['));
    } else if (template_strip && name.back() == '>') {
        name = name.substr(0, name.find('<'));
    }

    if (const_strip) {
        if (template_strip)
            erase_const(name);
        else
            name = remove_const(name);
    }
    return name;
}

//----------------------------------------------------------------------------
void CPyCppyy::TypeManip::cppscope_to_pyscope(std::string& cppscope)
{
// Change '::' in C++ scope into '.' as in a Python scope.
    std::string::size_type pos = 0;
    while ((pos = cppscope.find("::", pos)) != std::string::npos) {
        cppscope.replace(pos, 2, ".");
        pos += 1;
    }
}
