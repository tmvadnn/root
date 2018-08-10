#ifndef CPYCPPYY_CONVERTERS_H
#define CPYCPPYY_CONVERTERS_H

// Standard
#include <string>


namespace CPyCppyy {

struct Parameter;
struct CallContext;

class Converter {
public:
    virtual ~Converter() {}

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr) = 0;
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address);
};

// create converter from fully qualified type
Converter* CreateConverter(const std::string& fullType, Long_t size = -1);


// converters for special cases (only here b/c of external use of StrictCppObjectConverter)
class VoidArrayConverter : public Converter {
public:
    VoidArrayConverter(bool keepControl = true) { fKeepControl = keepControl; }

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address);

protected:
    virtual bool GetAddressSpecialCase(PyObject* pyobject, void*& address);
    bool KeepControl() { return fKeepControl; }

private:
    bool fKeepControl;
};

class CppObjectConverter : public VoidArrayConverter {
public:
    CppObjectConverter(Cppyy::TCppType_t klass, bool keepControl = false) :
        VoidArrayConverter(keepControl), fClass(klass) {}

public:
    virtual bool SetArg(PyObject*, Parameter&, CallContext* = nullptr);
    virtual PyObject* FromMemory(void* address);
    virtual bool ToMemory(PyObject* value, void* address);
        
protected:
    Cppyy::TCppType_t fClass;
};

class StrictCppObjectConverter : public CppObjectConverter {
public:
    using CppObjectConverter::CppObjectConverter;

protected:
    virtual bool GetAddressSpecialCase(PyObject*, void*&) { return false; }
};

} // namespace CPyCppyy

#endif // !CPYCPPYY_CONVERTERS_H
