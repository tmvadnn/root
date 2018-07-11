// Bindings
#include "CPyCppyy.h"
#include "ProxyWrappers.h"
#include "CPPClassMethod.h"
#include "CPPConstructor.h"
#include "CPPDataMember.h"
#include "CPPFunction.h"
#include "CPPInstance.h"
#include "CPPMethod.h"
#include "CPPOverload.h"
#include "CPPScope.h"
#include "CPPSetItem.h"
#include "MemoryRegulator.h"
#include "PyStrings.h"
#include "Pythonize.h"
#include "TemplateProxy.h"
#include "TupleOfInstances.h"
#include "TypeManip.h"
#include "Utility.h"

// Standard
#include <algorithm>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>


//- data _______________________________________________________________________
namespace CPyCppyy {
    extern PyObject* gThisModule;
    extern std::set<Cppyy::TCppType_t> gPinnedTypes;
}

// to prevent having to walk scopes, track python classes by C++ class
typedef std::map<Cppyy::TCppScope_t, PyObject*> PyClassMap_t;
static PyClassMap_t gPyClasses;


//- helpers --------------------------------------------------------------------

namespace CPyCppyy {

typedef struct {
    PyObject_HEAD
    PyObject *dict;
} proxyobject;

// helper for creating new C++ proxy python types
static PyObject* CreateNewCppProxyClass(Cppyy::TCppScope_t klass, PyObject* pybases)
{
// Create a new python shadow class with the required hierarchy and meta-classes.
    Py_XINCREF(pybases);
    if (!pybases) {
        pybases = PyTuple_New(1);
        Py_INCREF((PyObject*)(void*)&CPPInstance_Type);
        PyTuple_SET_ITEM(pybases, 0, (PyObject*)(void*)&CPPInstance_Type);
    }

    PyObject* pymetabases = PyTuple_New(PyTuple_GET_SIZE(pybases));
    for (int i = 0; i < PyTuple_GET_SIZE(pybases); ++i) {
        PyObject* btype = (PyObject*)Py_TYPE(PyTuple_GetItem(pybases, i));
        Py_INCREF(btype);
        PyTuple_SET_ITEM(pymetabases, i, btype);
    }

    std::string name = Cppyy::GetFinalName(klass);

// create meta-class, add a dummy __module__ to pre-empt the default setting
    PyObject* args = Py_BuildValue((char*)"sO{}", (name+"_meta").c_str(), pymetabases);
    PyDict_SetItem(PyTuple_GET_ITEM(args, 2), PyStrings::gModule, Py_True);
    Py_DECREF(pymetabases);

    PyObject* pymeta = PyType_Type.tp_new(&CPPScope_Type, args, nullptr);

    Py_DECREF(args);
    if (!pymeta) {
        PyErr_Print();
        Py_DECREF(pybases);
        return nullptr;
    }

// set the klass id, in case there is derivation Python-side
    ((CPPClass*)pymeta)->fCppType = klass;

// alright, and now we really badly want to get rid of the dummy ...
    PyObject* dictproxy = PyObject_GetAttr(pymeta, PyStrings::gDict);
    PyDict_DelItem(((proxyobject*)dictproxy)->dict, PyStrings::gModule);

// create actual class
    args = Py_BuildValue((char*)"sO{}", name.c_str(), pybases);
    PyObject* pyclass =
        ((PyTypeObject*)pymeta)->tp_new((PyTypeObject*)pymeta, args, nullptr);

    Py_DECREF(args);
    Py_DECREF(pymeta);
    Py_DECREF(pybases);

    return pyclass;
}

static inline
void AddPropertyToClass(PyObject* pyclass,
    Cppyy::TCppScope_t scope, Cppyy::TCppIndex_t idata)
{
    CPyCppyy::CPPDataMember* property = CPyCppyy::CPPDataMember_New(scope, idata);

// allow access at the instance level
    PyObject_SetAttrString(pyclass,
        const_cast<char*>(property->GetName().c_str()), (PyObject*)property);

// allow access at the class level (always add after setting instance level)
    if (Cppyy::IsStaticData(scope, idata)) {
        PyObject_SetAttrString((PyObject*)Py_TYPE(pyclass),
            const_cast<char*>(property->GetName().c_str()), (PyObject*)property);
    }

// done
    Py_DECREF(property);
}

} // namespace CPyCppyy


//- public functions ---------------------------------------------------------
namespace CPyCppyy {

static int BuildScopeProxyDict(Cppyy::TCppScope_t scope, PyObject* pyclass) {
// Collect methods and data for the given scope, and add them to the given python
// proxy object.

// some properties that'll affect building the dictionary
    bool isNamespace = Cppyy::IsNamespace(scope);
    bool hasConstructor = false;

// load all public methods and data members
    typedef std::vector<PyCallable*> Callables_t;
    typedef std::map<std::string, Callables_t> CallableCache_t;
    CallableCache_t cache;

// bypass custom __getattr__ for efficiency
    getattrofunc oldgetattro = Py_TYPE(pyclass)->tp_getattro;
    Py_TYPE(pyclass)->tp_getattro = PyType_Type.tp_getattro;

// functions in namespaces are properly found through lazy lookup, so do not
// create them until needed (the same is not true for data members)
    const Cppyy::TCppIndex_t nMethods =
        Cppyy::IsNamespace(scope) ? 0 : Cppyy::GetNumMethods(scope);
    for (Cppyy::TCppIndex_t imeth = 0; imeth < nMethods; ++imeth) {
        Cppyy::TCppMethod_t method = Cppyy::GetMethod(scope, imeth);

    // process the method based on its name
        std::string mtCppName = Cppyy::GetMethodName(method);

    // special case trackers
        bool setupSetItem = false;
        bool isConstructor = Cppyy::IsConstructor(method);
        bool isTemplate = isConstructor ? false : Cppyy::IsMethodTemplate(scope, imeth);

    // filter empty names (happens for namespaces, is bug?)
        if (mtCppName == "")
            continue;

    // filter C++ destructors
        if (mtCppName[0] == '~')
            continue;

    // translate operators
        std::string mtName = Utility::MapOperatorName(mtCppName, Cppyy::GetMethodNumArgs(method));

    // operator[]/() returning a reference type will be used for __setitem__
        if (mtName == "__call__" || mtName == "__getitem__") {
            const std::string& qual_return = Cppyy::ResolveName(Cppyy::GetMethodResultType(method));
            if (qual_return.find("const", 0, 5) == std::string::npos) {
                const std::string& cpd = Utility::Compound(qual_return);
                if (!cpd.empty() && cpd[cpd.size()- 1] == '&') {
                    setupSetItem = true;
                }
            }
        }

    // public methods are normally visible, private methods are mangled python-wise
    // note the overload implications which are name based, and note that genreflex
    // does not create the interface methods for private/protected methods ...
    // TODO: check whether Cling allows private method calling; otherwise delete this
        if (!Cppyy::IsPublicMethod(method)) {
            if (isConstructor)              // don't expose private ctors
                continue;
            else {                           // mangle private methods
            // TODO: drop use of TClassEdit here ...
            //            const std::string& clName = TClassEdit::ShortType(
            //               Cppyy::GetFinalName(scope).c_str(), TClassEdit::kDropAlloc);
                const std::string& clName = Cppyy::GetFinalName(scope);
                mtName = "_" + clName + "__" + mtName;
           }
       }

    // template members; handled by adding a dispatcher to the class
        if (isTemplate) {
        // TODO: the following is incorrect if both base and derived have the same
        // templated method (but that is an unlikely scenario anyway)
            PyObject* attr = PyObject_GetAttrString(pyclass, const_cast<char*>(mtName.c_str()));
            if (!TemplateProxy_Check(attr)) {
                PyErr_Clear();
                TemplateProxy* pytmpl = TemplateProxy_New(mtCppName, mtName, pyclass);
                if (CPPOverload_Check(attr)) pytmpl->AddOverload((CPPOverload*)attr);
                PyObject_SetAttrString(
                    pyclass, const_cast<char*>(mtName.c_str()), (PyObject*)pytmpl);
                Py_DECREF(pytmpl);
            }
            Py_XDECREF(attr);
        // continue processing to actually add the method so that the proxy can find
        // it on the class when called explicitly
        }

    // construct the holder
        PyCallable* pycall = 0;
        if (Cppyy::IsStaticMethod(method))  // class method
            pycall = new CPPClassMethod(scope, method);
        else if (isNamespace)               // free function
            pycall = new CPPFunction(scope, method);
        else if (isConstructor) {           // constructor
            pycall = new CPPConstructor(scope, method);
            mtName = "__init__";
            hasConstructor = true;
        } else                              // member function
            pycall = new CPPMethod(scope, method);

    // lookup method dispatcher and store method
        Callables_t& md = (*(cache.insert(
            std::make_pair(mtName, Callables_t())).first)).second;
        md.push_back(pycall);

    // special case for operator[]/() that returns by ref, use for getitem/call and setitem
        if (setupSetItem) {
            Callables_t& setitem = (*(cache.insert(
                std::make_pair(std::string("__setitem__"), Callables_t())).first)).second;
            setitem.push_back(new CPPSetItem(scope, method));
        }

        if (isTemplate) {
            PyObject* attr = PyObject_GetAttrString(pyclass, const_cast<char*>(mtName.c_str()));
            ((TemplateProxy*)attr)->AddTemplate(pycall->Clone());
            Py_DECREF(attr);
        }
    }

// add a pseudo-default ctor, if none defined
    if (!isNamespace && !hasConstructor)
        cache["__init__"].push_back(new CPPConstructor(scope, (Cppyy::TCppMethod_t)0));

// add the methods to the class dictionary
    for (CallableCache_t::iterator imd = cache.begin(); imd != cache.end(); ++imd) {
    // in order to prevent removing templated editions of this method (which were set earlier,
    // above, as a different proxy object), we'll check and add this method flagged as a generic
    // one (to be picked up by the templated one as appropriate) if a template exists
        PyObject* attr = PyObject_GetAttrString(pyclass, const_cast<char*>(imd->first.c_str()));
        if (TemplateProxy_Check(attr)) {
        // template exists, supply it with the non-templated method overloads
            for (Callables_t::iterator cit = imd->second.begin(); cit != imd->second.end(); ++cit)
                ((TemplateProxy*)attr)->AddOverload(*cit);
        } else {
            if (!attr) PyErr_Clear();
        // normal case, add a new method
            CPPOverload* method = CPPOverload_New(imd->first, imd->second);
            PyObject_SetAttrString(
                pyclass, const_cast<char*>(method->GetName().c_str()), (PyObject*)method);
            Py_DECREF(method);
        }

        Py_XDECREF(attr);         // could have be found in base class or non-existent
    }

 // collect data members (including enums)
    const Cppyy::TCppIndex_t nDataMembers = Cppyy::GetNumDatamembers(scope);
    for (Cppyy::TCppIndex_t idata = 0; idata < nDataMembers; ++idata) {
    // allow only public members
        if (!Cppyy::IsPublicData(scope, idata))
            continue;

    // enum datamembers (this in conjunction with previously collected enums above)
        if (Cppyy::IsEnumData(scope, idata) && Cppyy::IsStaticData(scope, idata)) {
        // some implementation-specific data members have no address: ignore them
            if (!Cppyy::GetDatamemberOffset(scope, idata))
                continue;

        // two options: this is a static variable, or it is the enum value, the latter
        // already exists, so check for it and move on if set
            PyObject* eset = PyObject_GetAttrString(pyclass,
                const_cast<char*>(Cppyy::GetDatamemberName(scope, idata).c_str()));
            if (eset) {
                Py_DECREF(eset);
                continue;
            }

            PyErr_Clear();

        // it could still be that this is an anonymous enum, which is not in the list
        // provided by the class
            if (strstr(Cppyy::GetDatamemberType(scope, idata).c_str(), "(anonymous)") != 0) {
                AddPropertyToClass(pyclass, scope, idata);
                continue;
            }
        }

    // properties (aka public (static) data members)
        AddPropertyToClass(pyclass, scope, idata);
    }

// restore custom __getattr__
    Py_TYPE(pyclass)->tp_getattro = oldgetattro;

// all ok, done
    return 0;
}

//----------------------------------------------------------------------------
static PyObject* BuildCppClassBases(Cppyy::TCppType_t klass)
{
// Build a tuple of python proxy classes of all the bases of the given 'klass'.

    size_t nbases = Cppyy::GetNumBases(klass);

// collect bases in acceptable mro order, while removing duplicates (this may
// break the overload resolution in esoteric cases, but otherwise the class can
// not be used at all, as CPython will refuse the mro).
    std::deque<std::string> uqb;
    std::deque<Cppyy::TCppType_t> bids;
    for (size_t ibase = 0; ibase < nbases; ++ibase) {
        const std::string& name = Cppyy::GetBaseName(klass, ibase);
        int decision = 2;
        Cppyy::TCppType_t tp = Cppyy::GetScope(name);
        for (size_t ibase2 = 0; ibase2 < uqb.size(); ++ibase2) {
            if (uqb[ibase2] == name) {         // not unique ... skip
                decision = 0;
                break;
            }

            if (Cppyy::IsSubtype(tp, bids[ibase2])) {
            // mro requirement: sub-type has to follow base
                decision = 1;
                break;
            }
        }

        if (decision == 1) {
            uqb.push_front(name);
            bids.push_front(tp);
        } else if (decision == 2) {
            uqb.push_back(name);
            bids.push_back(tp);
        }
    // skipped if decision == 0 (not unique)
    }

// allocate a tuple for the base classes, special case for first base
    nbases = uqb.size();

    PyObject* pybases = PyTuple_New(nbases ? nbases : 1);
    if (!pybases)
        return nullptr;

// build all the bases
    if (nbases == 0) {
        Py_INCREF((PyObject*)(void*)&CPPInstance_Type);
        PyTuple_SET_ITEM(pybases, 0, (PyObject*)(void*)&CPPInstance_Type);
    } else {
        for (std::vector<std::string>::size_type ibase = 0; ibase < nbases; ++ibase) {
            PyObject* pyclass = CreateScopeProxy(uqb[ibase]);
            if (!pyclass) {
                Py_DECREF(pybases);
                return nullptr;
            }

            PyTuple_SET_ITEM(pybases, ibase, pyclass);
        }

    // special case, if true python types enter the hierarchy, make sure that
    // the first base seen is still the CPPInstance_Type
        if (!PyObject_IsSubclass(PyTuple_GET_ITEM(pybases, 0), (PyObject*)&CPPInstance_Type)) {
            PyObject* newpybases = PyTuple_New(nbases+1);
            Py_INCREF((PyObject*)(void*)&CPPInstance_Type);
            PyTuple_SET_ITEM(newpybases, 0, (PyObject*)(void*)&CPPInstance_Type);
            for (int ibase = 0; ibase < (int)nbases; ++ibase) {
                PyObject* pyclass = PyTuple_GET_ITEM(pybases, ibase);
                Py_INCREF(pyclass);
                PyTuple_SET_ITEM(newpybases, ibase+1, pyclass);
            }
            Py_DECREF(pybases);
            pybases = newpybases;
        }
    }

    return pybases;
}

} // namespace CPyCppyy

//----------------------------------------------------------------------------
PyObject* CPyCppyy::GetScopeProxy(Cppyy::TCppScope_t scope)
{
// Retrieve scope proxy from the known ones.
    PyClassMap_t::iterator pci = gPyClasses.find(scope);
    if (pci != gPyClasses.end()) {
        PyObject* pyclass = PyWeakref_GetObject(pci->second);
        if (pyclass != Py_None) {
            Py_INCREF(pyclass);
            return pyclass;
        }
    }

    return nullptr;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CreateScopeProxy(Cppyy::TCppScope_t scope)
{
// Convenience function with a lookup first through the known existing proxies.
    PyObject* pyclass = GetScopeProxy(scope);
    if (pyclass)
        return pyclass;

    return CreateScopeProxy(Cppyy::GetScopedFinalName(scope));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CreateScopeProxy(PyObject*, PyObject* args)
{
// Build a python shadow class for the named C++ class.
    std::string cname = CPyCppyy_PyUnicode_AsString(PyTuple_GetItem(args, 0));
    if (PyErr_Occurred())
        return nullptr;

   return CreateScopeProxy(cname);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CreateScopeProxy(const std::string& scope_name, PyObject* parent)
{
// Build a python shadow class for the named C++ class or namespace.

// force building of the class if a parent is specified (prevents loops)
    bool force = parent != 0;

// working copy
    std::string name = scope_name;

// determine complete scope name, if a python parent has been given
    std::string scName = "";
    if (parent) {
        if (CPPScope_Check(parent))
            scName = Cppyy::GetScopedFinalName(((CPPScope*)parent)->fCppType);
        else {
            PyObject* parname = PyObject_GetAttr(parent, PyStrings::gName);
            if (!parname) {
                PyErr_Format(PyExc_SystemError, "given scope has no name for %s", name.c_str());
                return nullptr;
            }

        // should be a string
            scName = CPyCppyy_PyUnicode_AsString(parname);
            Py_DECREF(parname);
            if (PyErr_Occurred())
                return nullptr;
        }

    // accept this parent scope and use it's name for prefixing
        Py_INCREF(parent);
    }

// retrieve C++ class (this verifies name, and is therefore done first)
    const std::string& lookup = scName.empty() ? name : (scName+"::"+name);
    Cppyy::TCppScope_t klass = Cppyy::GetScope(lookup);

    if (!(bool)klass && Cppyy::IsTemplate(lookup)) {
    // a "naked" templated class is requested: return callable proxy for instantiations
        PyObject* pytcl = PyObject_GetAttr(gThisModule, PyStrings::gTemplate);
        PyObject* pytemplate = PyObject_CallFunction(
            pytcl, const_cast<char*>("s"), const_cast<char*>(lookup.c_str()));
        Py_DECREF(pytcl);

    // cache the result
        PyObject_SetAttrString(parent ? parent : gThisModule, (char*)name.c_str(), pytemplate);

    // done, next step should be a call into this template
        Py_XDECREF(parent);
        return pytemplate;
    }

    if (!(bool)klass) {   // if so, all options have been exhausted: it doesn't exist as such
        PyErr_Format(PyExc_TypeError, "\'%s\' is not a known C++ class", lookup.c_str());
        Py_XDECREF(parent);
        return nullptr;
    }

// locate class by ID, if possible, to prevent parsing scopes/templates anew
    PyObject* pyscope = GetScopeProxy(klass);
    if (pyscope) {
        if (parent) PyObject_SetAttrString(parent, (char*)scope_name.c_str(), pyscope);
        return pyscope;
    }

// locate the parent, if necessary, for building the class if not specified
    std::string::size_type last = 0;
    if (!parent) {
    // TODO: move this to TypeManip
    // need to deal with template paremeters that can have scopes themselves
        int tpl_open = 0;
        for (std::string::size_type pos = 0; pos < name.size(); ++pos) {
            std::string::value_type c = name[pos];

        // count '<' and '>' to be able to skip template contents
            if (c == '<')
                ++tpl_open;
            else if (c == '>')
                --tpl_open;

      // by only checking for "::" the last part (class name) is dropped
            else if (tpl_open == 0 && \
                c == ':' && pos+1 < name.size() && name[ pos+1 ] == ':') {
            // found a new scope part
                const std::string& part = name.substr(last, pos-last);

                PyObject* next = PyObject_GetAttrString(
                    parent ? parent : gThisModule, const_cast<char*>(part.c_str()));

                if (!next) {            // lookup failed, try to create it
                    PyErr_Clear();
                    next = CreateScopeProxy(part, parent);
                }
                Py_XDECREF(parent);

                if (!next)              // create failed, give up
                    return nullptr;

            // found scope part
                parent = next;

            // done with part (note that pos is moved one ahead here)
                last = pos+2; ++pos;
            }

        }

        if (parent && !CPPScope_Check(parent)) {
        // Special case: parent found is not one of ours (it's e.g. a pure Python module), so
        // continuing would fail badly. One final lookup, then out of here ...
            std::string unscoped = scope_name.substr(last, std::string::npos);
            return PyObject_GetAttrString(parent, unscoped.c_str());
        }
    }

// use the module as a fake cope if no outer scope found
    if (!parent) {
        parent = gThisModule;
        Py_INCREF(parent);
    }

// use actual class name for binding
    const std::string& actual = Cppyy::GetFinalName(klass);

// first try to retrieve an existing class representation
    PyObject* pyactual = CPyCppyy_PyUnicode_FromString(actual.c_str());
    PyObject* pyclass = force ? nullptr : PyObject_GetAttr(parent, pyactual);

    bool bClassFound = (bool)pyclass;

// build if the class does not yet exist
    if (!pyclass) {
    // ignore error generated from the failed lookup
        PyErr_Clear();

    // construct the base classes
        PyObject* pybases = BuildCppClassBases(klass);
        if (pybases != 0) {
        // create a fresh Python class, given bases, name, and empty dictionary
            pyclass = CreateNewCppProxyClass(klass, pybases);
            Py_DECREF(pybases);
        }

    // fill the dictionary, if successful
        if (pyclass) {
            if (BuildScopeProxyDict(klass, pyclass)) {
            // something failed in building the dictionary
                Py_DECREF(pyclass);
                pyclass = nullptr;
            } else {
                PyObject_SetAttr(parent, pyactual, pyclass);
            }
        }
    }

// give up, if not constructed
    if (!pyclass)
        return nullptr;

    if (name != actual)       // exists, but typedef-ed: simply map reference
        PyObject_SetAttrString(parent, const_cast<char*>(name.c_str()), pyclass);

// if this was a recycled class, we're done
    if (bClassFound)
        return pyclass;

// store a ref from cppyy scope id to new python class
    gPyClasses[klass] = PyWeakref_NewRef(pyclass, nullptr);

    Py_DECREF(pyactual);
    Py_DECREF(parent);

    if (!Cppyy::IsNamespace(klass)) {
    // add python-style features to classes
        if (!Pythonize(pyclass, Cppyy::GetScopedFinalName(klass))) {
            Py_XDECREF(pyclass);
            pyclass = nullptr;
        }
    } else {
    // add to sys.modules to allow importing from this namespace
        PyObject* pyfullname = PyObject_GetAttr(pyclass, PyStrings::gModule);
        CPyCppyy_PyUnicode_AppendAndDel(
            &pyfullname, CPyCppyy_PyUnicode_FromString("."));
        CPyCppyy_PyUnicode_AppendAndDel(
            &pyfullname, PyObject_GetAttr(pyclass, PyStrings::gName));
        PyObject* modules = PySys_GetObject(const_cast<char*>("modules"));
        if (modules && PyDict_Check(modules))
            PyDict_SetItem(modules, pyfullname, pyclass);
        Py_DECREF(pyfullname);
    }

// all done
    return pyclass;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::BindCppObjectNoCast(Cppyy::TCppObject_t address,
        Cppyy::TCppType_t klass, int flags)
{
// only known or knowable objects will be bound (null object is ok)
    if (!klass) {
        PyErr_SetString(PyExc_TypeError, "attempt to bind C++ object w/o class");
        return nullptr;
    }

// retrieve python class
    PyObject* pyclass = CreateScopeProxy(klass);
    if (!pyclass)
        return nullptr;                 // error has been set in CreateScopeProxy

    bool isRef   = flags & CPPInstance::kIsReference;
    bool isValue = flags & CPPInstance::kIsValue;

// TODO: add convenience function to MemoryRegulator to use pyclass directly
// TODO: make sure that a consistent address is used (may have to be done in BindCppObject)
    if (address && !isValue /* always fresh */) {
        PyObject* oldPyObject = MemoryRegulator::RetrievePyObject(
            isRef ? *(void**)address : address, klass);
    // ptr-ptr requires old object to be a reference to enable re-use
        if (oldPyObject && (!(flags & CPPInstance::kIsPtrPtr) ||
                ((CPPInstance*)oldPyObject)->fFlags & CPPInstance::kIsReference))
            return oldPyObject;
    }

// instantiate an object of this class
    PyObject* args = PyTuple_New(0);
    CPPInstance* pyobj =
        (CPPInstance*)((PyTypeObject*)pyclass)->tp_new((PyTypeObject*)pyclass, args, nullptr);
    Py_DECREF(args);
    Py_DECREF(pyclass);

// bind, register and return if successful
    if (pyobj != 0) { // fill proxy value?
        unsigned flags =
            (isRef ? CPPInstance::kIsReference : 0) | (isValue ? CPPInstance::kIsValue : 0);
        pyobj->Set(address, (CPPInstance::EFlags)flags);

        if (address && !isRef)
            MemoryRegulator::RegisterPyObject(pyobj, address);
    }

// successful completion
    return (PyObject*)pyobj;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::BindCppObject(Cppyy::TCppObject_t address,
        Cppyy::TCppType_t klass, int flags)
{
// if the object is a null pointer, return a typed one (as needed for overloading)
    if (!address)
        return BindCppObjectNoCast(address, klass, flags);

// only known or knowable objects will be bound
    if (!klass) {
        PyErr_SetString(PyExc_TypeError, "attempt to bind C++ object w/o class");
        return nullptr;
    }

    bool isRef = flags & CPPInstance::kIsReference;

// get actual class for recycling checking and/or downcasting
    Cppyy::TCppType_t clActual = isRef ? 0 : Cppyy::GetActualClass(klass, address);

// downcast to real class for object returns, unless pinned
    if (clActual && klass != clActual) {
        auto pci = gPinnedTypes.find(klass);
        if (pci == gPinnedTypes.end()) {
            ptrdiff_t offset = Cppyy::GetBaseOffset(
                clActual, klass, address, -1 /* down-cast */, true /* report errors */);
            if (offset != -1) {   // may fail if clActual not fully defined
                address = (void*)((Long_t)address + offset);
                klass = clActual;
            }
        }
    }

// actual binding (returned object may be zero w/ a python exception set)
    return BindCppObjectNoCast(address, klass, flags);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::BindCppObjectArray(
    Cppyy::TCppObject_t address, Cppyy::TCppType_t klass, Int_t size)
{
// TODO: this function exists for symmetry; need to figure out if it's useful
    return TupleOfInstances_New(address, klass, size);
}
