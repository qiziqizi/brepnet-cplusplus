#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <cassert>
#include <initializer_list>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <type_traits>
#include <map>
#include <memory>
#include <tuple>
#include <stdexcept>
#include <utility>
#include <limits>

namespace breptorch {

    enum DType { kFloat32, kInt, kLong };

    struct Tensor;
    template<typename T, int N> struct Accessor;

    // --------------------------------------------------------------------------
    // Storage (New: Holds actual data)
    // --------------------------------------------------------------------------
    struct Storage {
        std::vector<float> dataf_;
        std::vector<int64_t> datal_;

        Storage() {}
        Storage(size_t n, DType dt) {
            if (dt == kFloat32) dataf_.resize(n, 0.0f);
            else datal_.resize(n, 0);
        }
        // Default copy constructor performs deep copy of vectors
        Storage(const Storage& other) = default;
    };

    // --------------------------------------------------------------------------
    // Tensor (Reference Semantics)
    // --------------------------------------------------------------------------
    struct Tensor {
        std::shared_ptr<Storage> storage_;
        std::vector<int64_t> sizes_;
        DType dtype_ = kFloat32;
        int64_t storage_offset_ = 0;

        Tensor() {
            storage_ = std::make_shared<Storage>();
        }

        // Construct from float scalar
        explicit Tensor(float v) : sizes_({ 1 }), dtype_(kFloat32) {
            storage_ = std::make_shared<Storage>(1, kFloat32);
            storage_->dataf_[0] = v;
        }

        Tensor(const std::vector<int64_t>& sizes, DType dt = kFloat32) {
            sizes_ = sizes; dtype_ = dt;
            int64_t n = 1; for (auto s : sizes) n *= s;
            storage_ = std::make_shared<Storage>(n, dt);
        }

        // ���Ӵ˹��캯��������ƥ�� initializer_list
        Tensor(std::initializer_list<int64_t> sizes, DType dt = kFloat32)
            : Tensor(std::vector<int64_t>(sizes.begin(), sizes.end()), dt)
        {
            // ί�й��죺�� initializer_list ת��Ϊ vector���ٵ������й��캯��
        }

        // Static factories
        static Tensor zeros(const std::vector<int64_t>& sizes, DType dt = kFloat32) { return Tensor(sizes, dt); }
        static Tensor zeros(std::initializer_list<int64_t> sizes, DType dt = kFloat32) { return zeros(std::vector<int64_t>(sizes), dt); }
        static Tensor zeros(const std::vector<int64_t>& sizes, const Tensor& opts) { return zeros(sizes, opts.dtype_); }

        static Tensor ones(const std::vector<int64_t>& sizes, DType dt = kFloat32) {
            Tensor t(sizes, dt);
            if (dt == kFloat32) std::fill(t.storage_->dataf_.begin(), t.storage_->dataf_.end(), 1.0f);
            else std::fill(t.storage_->datal_.begin(), t.storage_->datal_.end(), 1);
            return t;
        }

        static Tensor eye(int n) {
            Tensor t({ (int64_t)n, (int64_t)n }, kFloat32);
            for (int i = 0; i < n; i++) t.at({ i,i }) = 1.0f;
            return t;
        }

        static Tensor full(const std::vector<int64_t>& sizes, float value, const Tensor& opts = Tensor()) {
            Tensor t(sizes, kFloat32); t.fill(value); return t;
        }

        // From Blob
        static Tensor from_blob(const float* data, const std::vector<int64_t>& shape, DType dt = kFloat32) {
            Tensor t(shape, dt); 
            std::memcpy(t.storage_->dataf_.data(), data, sizeof(float) * t.numel()); 
            return t;
        }
        static Tensor from_blob(const long long* data, const std::vector<int64_t>& shape, DType dt = kLong) {
            Tensor t(shape, dt); 
            int64_t n = t.numel(); 
            for (int i = 0; i < n; i++) t.storage_->datal_[i] = data[i]; 
            return t;
        }
        static Tensor from_blob(const int* data, const std::vector<int64_t>& shape, DType dt = kInt) {
            Tensor t(shape, kLong); 
            int64_t n = t.numel(); 
            for (int i = 0; i < n; i++) t.storage_->datal_[i] = data[i]; 
            return t;
        }

        // Properties
        bool defined() const { return !sizes_.empty() && storage_; }
        int64_t numel() const { if (sizes_.empty()) return 0; int64_t n = 1; for (auto s : sizes_) n *= s; return n; }
        const std::vector<int64_t>& sizes() const { return sizes_; }
        std::vector<int64_t> vec() const { return sizes_; }

        int64_t size(int i) const { return (i >= 0 && i < sizes_.size()) ? sizes_[i] : 0; }
        DType dtype() const { return dtype_; }
        int dim() const { return (int)sizes_.size(); }
        Tensor options() const { Tensor t; t.dtype_ = dtype_; return t; }

        // Accessors
        int64_t linear_idx(const std::vector<int64_t>& idx) const {
            int64_t off = 0, s = 1; for (int i = (int)sizes_.size() - 1; i >= 0; --i) { off += idx[i] * s; s *= sizes_[i]; } return off;
        }
        float& at(const std::vector<int64_t>& idx) { return storage_->dataf_[linear_idx(idx)]; }
        float at(const std::vector<int64_t>& idx) const { return storage_->dataf_[linear_idx(idx)]; }
        float& at(int64_t i, int64_t j) { return storage_->dataf_[i * sizes_[1] + j]; }
        float& at(int64_t i, int64_t j, int64_t k) { return storage_->dataf_[i * sizes_[1] * sizes_[2] + j * sizes_[2] + k]; }
        float& at(int64_t i, int64_t j, int64_t k, int64_t l) { return storage_->dataf_[i * sizes_[1] * sizes_[2] * sizes_[3] + j * sizes_[2] * sizes_[3] + k * sizes_[3] + l]; }

        // Ops
        // Clone: Deep copy
        Tensor clone() const { 
            Tensor t;
            t.sizes_ = sizes_;
            t.dtype_ = dtype_;
            if (storage_) t.storage_ = std::make_shared<Storage>(*storage_); // Deep copy storage
            else t.storage_ = std::make_shared<Storage>();
            return t; 
        }
        
        Tensor& fill(float v) { 
            if (dtype_ == kFloat32) std::fill(storage_->dataf_.begin(), storage_->dataf_.end(), v);
            else std::fill(storage_->datal_.begin(), storage_->datal_.end(), (int64_t)v);
            return *this; 
        }
        
        void zero_() { 
            if (dtype_ == kFloat32) std::fill(storage_->dataf_.begin(), storage_->dataf_.end(), 0.f); 
            else std::fill(storage_->datal_.begin(), storage_->datal_.end(), 0); 
        }

        Tensor to(DType dt) const {
            if (dtype_ == dt) return clone();
            Tensor t(sizes_, dt);
            if (dt == kLong) for (size_t i = 0; i < storage_->dataf_.size(); i++) t.storage_->datal_[i] = (int64_t)storage_->dataf_[i];
            else for (size_t i = 0; i < storage_->datal_.size(); i++) t.storage_->dataf_[i] = (float)storage_->datal_[i];
            return t;
        }

        Tensor& sub_(const Tensor& b) { 
            for (size_t i = 0; i < storage_->dataf_.size(); i++) storage_->dataf_[i] -= b.storage_->dataf_[i]; 
            return *this; 
        }
        Tensor& div_(const Tensor& b) { 
            for (size_t i = 0; i < storage_->dataf_.size(); i++) storage_->dataf_[i] /= b.storage_->dataf_[i]; 
            return *this; 
        }
        
        void index_put_(const std::initializer_list<int64_t>& idx, float v) {
            if (idx.size() == 1) {
                int64_t r = *idx.begin();
                int64_t row_size = 1;
                for(size_t i=1; i<sizes_.size(); ++i) row_size *= sizes_[i];
                
                size_t start = r * row_size;
                size_t end = start + row_size;
                if (dtype_ == kFloat32) {
                    if (start < storage_->dataf_.size() && end <= storage_->dataf_.size())
                        for(size_t i=start; i<end; ++i) storage_->dataf_[i] = v;
                } else {
                    if (start < storage_->datal_.size() && end <= storage_->datal_.size())
                        for(size_t i=start; i<end; ++i) storage_->datal_[i] = (int64_t)v;
                }
            }
        }

        Tensor flatten(int start_dim = 0) const {
            if (start_dim < 0) start_dim += (int)sizes_.size();
            std::vector<int64_t> new_sizes;
            for (int i = 0; i < start_dim; ++i) new_sizes.push_back(sizes_[i]);
            int64_t flat_size = 1;
            for (size_t i = start_dim; i < sizes_.size(); ++i) flat_size *= sizes_[i];
            new_sizes.push_back(flat_size);
            return view(new_sizes);
        }

        // View: Shallow copy (share storage)
        Tensor view(const std::vector<int64_t>& s) const {
            Tensor t;
            t.storage_ = storage_; // Share storage
            t.dtype_ = dtype_;
            
            std::vector<int64_t> new_sizes = s;
            int64_t total = numel();
            int infer_idx = -1;
            int64_t known = 1;
            for (size_t i = 0; i < s.size(); ++i) {
                if (s[i] == -1) {
                    if (infer_idx != -1) throw std::runtime_error("Only one dimension can be inferred");
                    infer_idx = (int)i;
                }
                else {
                    known *= s[i];
                }
            }
            if (infer_idx != -1) {
                if (known == 0) new_sizes[infer_idx] = 0;
                else new_sizes[infer_idx] = total / known;
            }
            t.sizes_ = new_sizes;
            return t;
        }
        Tensor reshape(const std::vector<int64_t>& s) const { return view(s); }
        
        // Slice: Deep copy for now
        Tensor slice(int dim, int64_t start, int64_t end) const {
            if (dim < 0) dim += (int)sizes_.size();
            if (start < 0) start += sizes_[dim];
            if (end < 0) end += sizes_[dim];
            if (end > sizes_[dim]) end = sizes_[dim];
            if (start >= end) return Tensor::zeros({0}, dtype_);

            std::vector<int64_t> new_sizes = sizes_;
            new_sizes[dim] = end - start;
            
            Tensor out(new_sizes, dtype_); // New storage
            
            std::vector<int64_t> strides(sizes_.size());
            int64_t s = 1;
            for(int i=(int)sizes_.size()-1; i>=0; --i) { strides[i] = s; s *= sizes_[i]; }
            
            if (dim == 0) {
                size_t copy_count = (end - start) * strides[0];
                size_t offset = start * strides[0];
                if (dtype_ == kFloat32) {
                    std::memcpy(out.storage_->dataf_.data(), storage_->dataf_.data() + offset, copy_count * sizeof(float));
                } else {
                    std::memcpy(out.storage_->datal_.data(), storage_->datal_.data() + offset, copy_count * sizeof(int64_t));
                }
                return out;
            }
            
            int64_t numel = out.numel();
            for(int64_t i=0; i<numel; ++i) {
                int64_t temp = i;
                int64_t src_idx = 0;
                for(int d=(int)new_sizes.size()-1; d>=0; --d) {
                    int64_t coord = temp % new_sizes[d];
                    temp /= new_sizes[d];
                    if (d == dim) coord += start;
                    src_idx += coord * strides[d];
                }
                if (dtype_ == kFloat32) out.storage_->dataf_[i] = storage_->dataf_[src_idx];
                else out.storage_->datal_[i] = storage_->datal_[src_idx];
            }
            return out;
        }

        // Copy_: Deep copy data into current storage
        void copy_(const Tensor& other) { 
            if (storage_ == other.storage_) return;
            if (!storage_) storage_ = std::make_shared<Storage>();
            
            if (dtype_ == kFloat32) {
                if (other.dtype_ == kFloat32) storage_->dataf_ = other.storage_->dataf_;
                else {
                    storage_->dataf_.resize(other.storage_->datal_.size());
                    for(size_t i=0; i<other.storage_->datal_.size(); ++i) storage_->dataf_[i] = (float)other.storage_->datal_[i];
                }
            } else {
                if (other.dtype_ == kLong) storage_->datal_ = other.storage_->datal_;
                else {
                    storage_->datal_.resize(other.storage_->dataf_.size());
                    for(size_t i=0; i<other.storage_->dataf_.size(); ++i) storage_->datal_[i] = (int64_t)other.storage_->dataf_[i];
                }
            }
        }

        Tensor index(const std::initializer_list<Tensor>& idxs) const {
            if (idxs.size() == 0) return *this;
            const Tensor& idx = *idxs.begin();
            
            std::vector<int64_t> new_sizes = idx.sizes_;
            for(size_t i=1; i<sizes_.size(); ++i) new_sizes.push_back(sizes_[i]);
            
            Tensor out(new_sizes, dtype_);
            
            int64_t row_size = 1;
            for(size_t i=1; i<sizes_.size(); ++i) row_size *= sizes_[i];
            
            const int64_t* idx_data = idx.storage_->datal_.data();
            
            for(int64_t i=0; i<idx.numel(); ++i) {
                int64_t src_row = idx_data[i];
                if (src_row >= 0 && src_row < sizes_[0]) {
                    if (dtype_ == kFloat32) {
                        std::memcpy(out.storage_->dataf_.data() + i * row_size, 
                                    storage_->dataf_.data() + src_row * row_size, 
                                    row_size * sizeof(float));
                    } else {
                        std::memcpy(out.storage_->datal_.data() + i * row_size, 
                                    storage_->datal_.data() + src_row * row_size, 
                                    row_size * sizeof(int64_t));
                    }
                }
            }
            return out;
        }

        Tensor select(int dim, int64_t idx) const { return clone(); /* TODO */ }

        template<typename T> T* data_ptr() { return (T*)(dtype_ == kFloat32 ? (void*)storage_->dataf_.data() : (void*)storage_->datal_.data()); }

        template<typename T> T item() const { return (T)(dtype_ == kFloat32 ? storage_->dataf_[0] : storage_->datal_[0]); }
        
        Tensor max() const { 
            if (numel() == 0) return Tensor(0.0f);
            if (dtype_ == kFloat32) {
                float m = -std::numeric_limits<float>::infinity();
                for(float x : storage_->dataf_) if(x > m) m = x;
                return Tensor(m);
            } else {
                int64_t m = -std::numeric_limits<int64_t>::max();
                for(int64_t x : storage_->datal_) if(x > m) m = x;
                return Tensor((float)m).to(kLong);
            }
        }
        Tensor min() const { 
            if (numel() == 0) return Tensor(0.0f);
            if (dtype_ == kFloat32) {
                float m = std::numeric_limits<float>::infinity();
                for(float x : storage_->dataf_) if(x < m) m = x;
                return Tensor(m);
            } else {
                int64_t m = std::numeric_limits<int64_t>::max();
                for(int64_t x : storage_->datal_) if(x < m) m = x;
                return Tensor((float)m).to(kLong);
            }
        }
        Tensor abs() const { 
            Tensor out = clone();
            if (dtype_ == kFloat32) {
                for(auto& x : out.storage_->dataf_) x = std::abs(x);
            } else {
                for(auto& x : out.storage_->datal_) x = std::abs(x);
            }
            return out;
        }
        Tensor sum() const { 
            if (dtype_ == kFloat32) {
                float s = 0;
                for(float x : storage_->dataf_) s += x;
                return Tensor(s);
            } else {
                int64_t s = 0;
                for(int64_t x : storage_->datal_) s += x;
                return Tensor((float)s); 
            }
        }

        // Operators
        Tensor operator-(const Tensor& b) const { 
            Tensor out = clone();
            if (dtype_ == kFloat32) {
                for(size_t i=0; i<out.storage_->dataf_.size(); ++i) out.storage_->dataf_[i] -= b.storage_->dataf_[i];
            } else {
                for(size_t i=0; i<out.storage_->datal_.size(); ++i) out.storage_->datal_[i] -= b.storage_->datal_[i];
            }
            return out;
        }
        Tensor operator+(const Tensor& b) const { 
            Tensor out = clone();
            if (dtype_ == kFloat32) {
                for(size_t i=0; i<out.storage_->dataf_.size(); ++i) out.storage_->dataf_[i] += b.storage_->dataf_[i];
            } else {
                for(size_t i=0; i<out.storage_->datal_.size(); ++i) out.storage_->datal_[i] += b.storage_->datal_[i];
            }
            return out;
        }
        Tensor operator*(float v) const { 
            Tensor out = clone();
            if (dtype_ == kFloat32) {
                for(size_t i=0; i<out.storage_->dataf_.size(); ++i) out.storage_->dataf_[i] *= v;
            } else {
                for(size_t i=0; i<out.storage_->datal_.size(); ++i) out.storage_->datal_[i] *= (int64_t)v;
            }
            return out;
        }
        Tensor operator*(const Tensor& b) const { 
            Tensor out = clone();
            if (dtype_ == kFloat32) {
                for(size_t i=0; i<out.storage_->dataf_.size(); ++i) out.storage_->dataf_[i] *= b.storage_->dataf_[i];
            } else {
                for(size_t i=0; i<out.storage_->datal_.size(); ++i) out.storage_->datal_[i] *= b.storage_->datal_[i];
            }
            return out;
        }
        Tensor operator+(float v) const { 
            Tensor out = clone();
            if (dtype_ == kFloat32) {
                for(size_t i=0; i<out.storage_->dataf_.size(); ++i) out.storage_->dataf_[i] += v;
            } else {
                for(size_t i=0; i<out.storage_->datal_.size(); ++i) out.storage_->datal_[i] += (int64_t)v;
            }
            return out;
        }
        Tensor operator/(const Tensor& b) const { 
            Tensor out = clone();
            if (dtype_ == kFloat32) {
                for(size_t i=0; i<out.storage_->dataf_.size(); ++i) out.storage_->dataf_[i] /= b.storage_->dataf_[i];
            } else {
                for(size_t i=0; i<out.storage_->datal_.size(); ++i) out.storage_->datal_[i] /= b.storage_->datal_[i];
            }
            return out;
        }

        Tensor operator/(float v) const { 
            Tensor out = clone();
            if (dtype_ == kFloat32) {
                for(size_t i=0; i<out.storage_->dataf_.size(); ++i) out.storage_->dataf_[i] /= v;
            } else {
                for(size_t i=0; i<out.storage_->datal_.size(); ++i) out.storage_->datal_[i] /= (int64_t)v;
            }
            return out;
        }
        Tensor operator<(float v) const { 
            Tensor out = Tensor(sizes_, kFloat32); 
            if (dtype_ == kFloat32) {
                for(size_t i=0; i<storage_->dataf_.size(); ++i) out.storage_->dataf_[i] = (storage_->dataf_[i] < v) ? 1.0f : 0.0f;
            } else {
                for(size_t i=0; i<storage_->datal_.size(); ++i) out.storage_->dataf_[i] = (storage_->datal_[i] < v) ? 1.0f : 0.0f;
            }
            return out;
        }

        template<typename T, int N> Accessor<T, N> accessor() { return Accessor<T, N>(this); }

        struct RowProxy { Tensor* t; int64_t r; void zero_() {} operator Tensor() const { return Tensor(); } };
        RowProxy operator[](int64_t i) { return RowProxy{ this, i }; }

        // Friend for printing
        friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
    };

    // Print implementation
    inline std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << "Tensor(";
        // Print data (flattened for simplicity)
        int64_t numel = t.numel();
        int64_t limit = 50; // Limit output
        
        os << "[";
        for(int64_t i=0; i<numel && i<limit; ++i) {
            if (t.dtype_ == kFloat32) os << t.storage_->dataf_[i];
            else os << t.storage_->datal_[i];
            if (i < numel-1 && i < limit-1) os << ", ";
        }
        if (numel > limit) os << ", ...";
        os << "]";
        
        // Print size
        os << ", size=(";
        for (size_t i = 0; i < t.sizes_.size(); ++i) {
            os << t.sizes_[i];
            if (i < t.sizes_.size() - 1) os << ", ";
        }
        os << "))";
        return os;
    }

    // --------------------------------------------------------------------------
    // Accessor
    // --------------------------------------------------------------------------
    template<> struct Accessor<float, 2> { Tensor* t; Accessor(Tensor* _t) :t(_t) {} struct Row { Tensor* t; int64_t i; float& operator[](int64_t j) { return t->at(i, j); } }; Row operator[](int64_t i) { return Row{ t,i }; } };
    template<> struct Accessor<float, 3> { Tensor* t; Accessor(Tensor* _t) :t(_t) {} struct Plane { Tensor* t; int64_t i; struct Row { Tensor* t; int64_t i, j; float& operator[](int64_t k) { return t->at(i, j, k); } }; Row operator[](int64_t j) { return Row{ t,i,j }; } }; Plane operator[](int64_t i) { return Plane{ t,i }; } };

    // --------------------------------------------------------------------------
    // Free Functions
    // --------------------------------------------------------------------------
    
    // Helper: Calculate strides
    inline std::vector<int64_t> calc_strides(const std::vector<int64_t>& sizes) {
        std::vector<int64_t> strides(sizes.size());
        if (sizes.empty()) return strides;
        int64_t s = 1;
        for (int i = (int)sizes.size() - 1; i >= 0; --i) {
            strides[i] = s;
            s *= sizes[i];
        }
        return strides;
    }

    // Static wrappers
    inline Tensor flip(const Tensor& t, std::initializer_list<int> d) { 
        if (d.size() == 0) return t.clone();
        int dim = *d.begin();
        if (dim < 0) dim += t.dim();
        
        Tensor out = t.clone();
        int64_t dim_size = t.size(dim);
        std::vector<int64_t> strides = calc_strides(t.sizes_);
        int64_t stride = strides[dim];
        int64_t size = t.sizes_[dim];
        int64_t numel = t.numel();
        
        if (t.dtype_ == kFloat32) {
            float* data = out.storage_->dataf_.data();
            for(int64_t i=0; i<numel; ++i) {
                int64_t temp = i;
                int64_t coord = 0;
                for(int k=(int)t.sizes_.size()-1; k>=0; --k) {
                    int64_t c = temp % t.sizes_[k];
                    temp /= t.sizes_[k];
                    if (k == dim) coord = c;
                }
                if (coord < size / 2) {
                    int64_t other_idx = i + (size - 1 - 2*coord) * stride;
                    std::swap(data[i], data[other_idx]);
                }
            }
        }
        return out;
    }

    inline Tensor matmul(const Tensor& a, const Tensor& b) {
        if (a.dim() != 2 || b.dim() != 2) throw std::runtime_error("matmul: only supports 2D tensors");
        if (a.size(1) != b.size(0)) throw std::runtime_error("matmul: shape mismatch");
        
        int64_t M = a.size(0);
        int64_t K = a.size(1);
        int64_t N = b.size(1);

        Tensor c({M, N}, a.dtype_);
        
        if (a.dtype_ == kFloat32) {
            const float* adata = a.storage_->dataf_.data();
            const float* bdata = b.storage_->dataf_.data();
            float* cdata = c.storage_->dataf_.data();
            
            for(int64_t i=0; i<M; ++i) {
                for(int64_t j=0; j<N; ++j) {
                    float sum = 0;
                    for(int64_t k=0; k<K; ++k) {
                        sum += adata[i*K + k] * bdata[k*N + j];
                    }
                    cdata[i*N + j] = sum;
                }
            }
        }
        return c;
    }

    inline float det(const Tensor& m) {
        if (m.dim() != 2 || m.size(0) != 4 || m.size(1) != 4) return 0.0f;
        const float* d = m.storage_->dataf_.data();
        float m00 = d[0],  m01 = d[1],  m02 = d[2],  m03 = d[3];
        float m10 = d[4],  m11 = d[5],  m12 = d[6],  m13 = d[7];
        float m20 = d[8],  m21 = d[9],  m22 = d[10], m23 = d[11];
        float m30 = d[12], m31 = d[13], m32 = d[14], m33 = d[15];

        float A2323 = m22 * m33 - m23 * m32;
        float A1323 = m21 * m33 - m23 * m31;
        float A1223 = m21 * m32 - m22 * m31;
        float A0323 = m20 * m33 - m23 * m30;
        float A0223 = m20 * m32 - m22 * m30;
        float A0123 = m20 * m31 - m21 * m30;

        return m00 * ( m11 * A2323 - m12 * A1323 + m13 * A1223 ) 
             - m01 * ( m10 * A2323 - m12 * A0323 + m13 * A0223 ) 
             + m02 * ( m10 * A1323 - m11 * A0323 + m13 * A0123 ) 
             - m03 * ( m10 * A1223 - m11 * A0223 + m12 * A0123 );
    }

    inline Tensor inverse(const Tensor& m) {
        if (m.dim() != 2 || m.size(0) != 4 || m.size(1) != 4) return m.clone();
        Tensor inv({4, 4}, kFloat32);
        const float* m_ = m.storage_->dataf_.data();
        float* inv_ = inv.storage_->dataf_.data();

        float m00 = m_[0],  m01 = m_[1],  m02 = m_[2],  m03 = m_[3];
        float m10 = m_[4],  m11 = m_[5],  m12 = m_[6],  m13 = m_[7];
        float m20 = m_[8],  m21 = m_[9],  m22 = m_[10], m23 = m_[11];
        float m30 = m_[12], m31 = m_[13], m32 = m_[14], m33 = m_[15];

        float v0 = m20 * m31 - m21 * m30;
        float v1 = m20 * m32 - m22 * m30;
        float v2 = m20 * m33 - m23 * m30;
        float v3 = m21 * m32 - m22 * m31;
        float v4 = m21 * m33 - m23 * m31;
        float v5 = m22 * m33 - m23 * m32;

        float t00 = + (v5 * m11 - v4 * m12 + v3 * m13);
        float t10 = - (v5 * m10 - v2 * m12 + v1 * m13);
        float t20 = + (v4 * m10 - v2 * m11 + v0 * m13);
        float t30 = - (v3 * m10 - v1 * m11 + v0 * m12);

        float invDet = t00 * m00 + t10 * m01 + t20 * m02 + t30 * m03;
        if (std::abs(invDet) < 1e-9) return m.clone();

        invDet = 1.0f / invDet;

        inv_[0] = t00 * invDet;
        inv_[4] = t10 * invDet;
        inv_[8] = t20 * invDet;
        inv_[12] = t30 * invDet;

        inv_[1] = - (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
        inv_[5] = + (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
        inv_[9] = - (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
        inv_[13] = + (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

        v0 = m10 * m31 - m11 * m30;
        v1 = m10 * m32 - m12 * m30;
        v2 = m10 * m33 - m13 * m30;
        v3 = m11 * m32 - m12 * m31;
        v4 = m11 * m33 - m13 * m31;
        v5 = m12 * m33 - m13 * m32;

        inv_[2] = + (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
        inv_[6] = - (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
        inv_[10] = + (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
        inv_[14] = - (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

        v0 = m21 * m10 - m20 * m11;
        v1 = m22 * m10 - m20 * m12;
        v2 = m23 * m10 - m20 * m13;
        v3 = m22 * m11 - m21 * m12;
        v4 = m23 * m11 - m21 * m13;
        v5 = m23 * m12 - m22 * m13;

        inv_[3] = - (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
        inv_[7] = + (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
        inv_[11] = - (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
        inv_[15] = + (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

        return inv;
    }

    inline Tensor where(const Tensor& c, const Tensor& a, const Tensor& b) { 
        Tensor out = a.clone();
        int64_t n = out.numel();
        if (out.dtype_ == kFloat32) {
            for(int64_t i=0; i<n; ++i) {
                if (c.storage_->dataf_[i] == 0.0f) out.storage_->dataf_[i] = b.storage_->dataf_[i];
            }
        }
        return out;
    }

    inline Tensor cross(const Tensor& a, const Tensor& b) {
        if (a.numel() != 3 || b.numel() != 3) return Tensor();
        Tensor c({3}, kFloat32);
        float a0 = a.at({0}), a1 = a.at({1}), a2 = a.at({2});
        float b0 = b.at({0}), b1 = b.at({1}), b2 = b.at({2});
        c.at({0}) = a1*b2 - a2*b1;
        c.at({1}) = a2*b0 - a0*b2;
        c.at({2}) = a0*b1 - a1*b0;
        return c;
    }

    inline Tensor flatten(const Tensor& t, int d = 0) { return t.flatten(d); }

    inline Tensor mean(const Tensor& t, int dim) {
        if (dim < 0) dim += t.dim();
        std::vector<int64_t> out_sizes = t.sizes_;
        out_sizes.erase(out_sizes.begin() + dim);
        if (out_sizes.empty()) out_sizes = {1};

        Tensor out(out_sizes, kFloat32);
        std::vector<int64_t> in_strides = calc_strides(t.sizes_);
        int64_t dim_stride = in_strides[dim];
        int64_t dim_size = t.sizes_[dim];
        int64_t out_numel = out.numel();

        for(int64_t i=0; i<out_numel; ++i) {
            int64_t temp = i;
            int64_t base_offset = 0;
            for(int d=(int)out_sizes.size()-1; d>=0; --d) {
                int64_t coord = temp % out_sizes[d];
                temp /= out_sizes[d];
                int input_d = (d < dim) ? d : d+1;
                base_offset += coord * in_strides[input_d];
            }
            
            float sum = 0;
            if (t.dtype_ == kFloat32) {
                for(int64_t k=0; k<dim_size; ++k) sum += t.storage_->dataf_[base_offset + k * dim_stride];
            } else {
                for(int64_t k=0; k<dim_size; ++k) sum += (float)t.storage_->datal_[base_offset + k * dim_stride];
            }
            out.storage_->dataf_[i] = sum / dim_size;
        }
        return out;
    }

    inline std::tuple<Tensor, Tensor> max(const Tensor& t, int dim) {
        if (dim < 0) dim += t.dim();
        std::vector<int64_t> out_sizes = t.sizes_;
        out_sizes.erase(out_sizes.begin() + dim);
        if (out_sizes.empty()) out_sizes = {1};

        Tensor values(out_sizes, t.dtype_);
        Tensor indices(out_sizes, kLong);

        std::vector<int64_t> in_strides = calc_strides(t.sizes_);
        int64_t dim_stride = in_strides[dim];
        int64_t dim_size = t.sizes_[dim];
        int64_t out_numel = values.numel();

        for(int64_t i=0; i<out_numel; ++i) {
            int64_t temp = i;
            int64_t base_offset = 0;
            for(int d=(int)out_sizes.size()-1; d>=0; --d) {
                int64_t coord = temp % out_sizes[d];
                temp /= out_sizes[d];
                int input_d = (d < dim) ? d : d+1;
                base_offset += coord * in_strides[input_d];
            }
            
            if (t.dtype_ == kFloat32) {
                float max_val = -std::numeric_limits<float>::infinity();
                int64_t max_idx = 0;
                for(int64_t k=0; k<dim_size; ++k) {
                    float val = t.storage_->dataf_[base_offset + k * dim_stride];
                    if (val > max_val) { max_val = val; max_idx = k; }
                }
                values.storage_->dataf_[i] = max_val;
                indices.storage_->datal_[i] = max_idx;
            } else {
                int64_t max_val = -std::numeric_limits<int64_t>::max();
                int64_t max_idx = 0;
                for(int64_t k=0; k<dim_size; ++k) {
                    int64_t val = t.storage_->datal_[base_offset + k * dim_stride];
                    if (val > max_val) { max_val = val; max_idx = k; }
                }
                values.storage_->datal_[i] = max_val;
                indices.storage_->datal_[i] = max_idx;
            }
        }
        return {values, indices};
    }

    //inline float dot(const Tensor& a, const Tensor& b) {
    //    float sum = 0;
    //    if (a.dtype_ == kFloat32) {
    //        for(size_t i=0; i<a.storage_->dataf_.size(); ++i) sum += a.storage_->dataf_[i] * b.storage_->dataf_[i];
    //    }
    //    return sum;
    //}

    //inline float norm(const Tensor& a) {
    //    float sum = 0;
    //    if (a.dtype_ == kFloat32) {
    //        for(float x : a.storage_->dataf_) sum += x*x;
    //    }
    //    return std::sqrt(sum);
    //}
    //
    inline Tensor dot(const Tensor& a, const Tensor& b) {
        // У�飺������ 3D ������ƥ����� ProjectVector ������
        if (a.sizes() != std::vector<int64_t>{3} || b.sizes() != std::vector<int64_t>{3}) {
            throw std::runtime_error("dot only supports 3D vectors (size={3})");
        }
        float sum = 0.0f;
        if (a.dtype_ == kFloat32 && b.dtype_ == kFloat32) {
            for (size_t i = 0; i < 3; ++i) { // �̶�3ά������Ч
                sum += a.storage_->dataf_[i] * b.storage_->dataf_[i];
            }
        }
        return Tensor(sum); // ���� Tensor��ƥ������� .item<float>() ���߼�
    }

    // 2. ��ȫ norm ������breptorch �����ռ��ڣ����� Tensor��
    inline Tensor norm(const Tensor& a) {
        if (a.sizes() != std::vector<int64_t>{3}) {
            throw std::runtime_error("norm only supports 3D vectors (size={3})");
        }
        float sum_sq = 0.0f;
        if (a.dtype_ == kFloat32) {
            for (size_t i = 0; i < 3; ++i) {
                sum_sq += a.storage_->dataf_[i] * a.storage_->dataf_[i];
            }
        }
        return Tensor(std::sqrt(sum_sq)); // ���� Tensor
    }

    inline Tensor cat(const std::vector<Tensor>& l, int d) {
        if (l.empty()) return Tensor();
        if (l.size() == 1) return l[0].clone();

        DType dt = l[0].dtype_;
        for (const auto& t : l) {
            if (t.dtype_ != dt) throw std::runtime_error("cat: all tensors must have same dtype");
        }

        std::vector<int64_t> out_sizes = l[0].sizes_;
        int64_t cat_dim_size = 0;
        for (const auto& t : l) cat_dim_size += t.size(d);
        out_sizes[d] = cat_dim_size;


        Tensor out(out_sizes, dt);

        if (d == 0) {
            size_t offset = 0;
            for (const auto& t : l) {
                size_t bytes = t.numel() * (t.dtype_ == kFloat32 ? sizeof(float) : sizeof(int64_t));
                if (t.dtype_ == kFloat32) {
                    std::memcpy(out.storage_->dataf_.data() + offset,
                        t.storage_->dataf_.data(), bytes);
                }
                else {
                    std::memcpy(out.storage_->datal_.data() + offset,
                        t.storage_->datal_.data(), bytes);
                }
                offset += t.numel();
            }
            return out;
        }

        int64_t outer_elements = 1;
        for (int i = 0; i < d; ++i) outer_elements *= out_sizes[i];

        int64_t inner_elements = 1;
        for (size_t i = d + 1; i < out_sizes.size(); ++i) inner_elements *= out_sizes[i];

        size_t element_size = (out.dtype_ == kFloat32 ? sizeof(float) : sizeof(int64_t));
        char* out_ptr = (char*)(out.dtype_ == kFloat32 ? (void*)out.storage_->dataf_.data() : (void*)out.storage_->datal_.data());

        for (int64_t i = 0; i < outer_elements; ++i) {
            for (const auto& t : l) {
                int64_t dim_d_size = t.size(d);
                size_t copy_bytes = dim_d_size * inner_elements * element_size;

                const char* src_ptr = (const char*)(t.dtype_ == kFloat32 ? (const void*)t.storage_->dataf_.data() : (const void*)t.storage_->datal_.data());
                src_ptr += i * dim_d_size * inner_elements * element_size;

                std::memcpy(out_ptr, src_ptr, copy_bytes);
                out_ptr += copy_bytes;
            }
        }

        return out;
    }


    inline Tensor stack(const std::vector<Tensor>& l, int dim = 0) {
        if (l.empty()) return Tensor();
        // 1. Unsqueeze each tensor at dim
        std::vector<Tensor> unsqueezed;
        unsqueezed.reserve(l.size());
        for(const auto& t : l) {
            std::vector<int64_t> new_shape = t.sizes_;
            if (dim < 0) dim += (int)new_shape.size() + 1;
            new_shape.insert(new_shape.begin() + dim, 1);
            unsqueezed.push_back(t.view(new_shape));
        }
        // 2. Cat
        return cat(unsqueezed, dim);
    }

    // Global wrappers
    inline Tensor zeros(std::initializer_list<int64_t> s, DType dt = kFloat32) { return Tensor::zeros(s, dt); }
    inline Tensor zeros(const std::vector<int64_t>& s, const Tensor& o) { return Tensor::zeros(s, o); } // Add vector overload
    inline Tensor ones(std::vector<int64_t> s) { return Tensor::ones(s); }
    inline Tensor ones(std::initializer_list<int64_t> s, DType dt = kFloat32) { return Tensor::ones(s, dt); }
    inline Tensor ones(std::vector<int64_t> s, const Tensor& o) { return Tensor::ones(s); }
    inline Tensor full(std::vector<int64_t> s, float v, const Tensor& o = Tensor()) { return Tensor::full(s, v); }
    inline Tensor from_blob(const float* d, std::vector<int64_t> s, DType dt = kFloat32) { return Tensor::from_blob(d, s, dt); }
    inline Tensor from_blob(const long long* d, std::vector<int64_t> s, DType dt = kLong) { return Tensor::from_blob(d, s, dt); }
    inline Tensor from_blob(const int* d, std::vector<int64_t> s, DType dt = kInt) { return Tensor::from_blob(d, s, dt); }
    inline Tensor tensor(std::initializer_list<float> v) { Tensor t(std::vector<int64_t>{ (int64_t)v.size() }); int i = 0; for (auto x : v) t.storage_->dataf_[i++] = x; return t; }
    inline Tensor tensor(std::initializer_list<float> v, const Tensor& o) { return tensor(v); }
    inline Tensor tensor(float v) { return Tensor(v); }
    inline Tensor ones_like(const Tensor& t) { return Tensor::ones(t.sizes_); }
    inline Tensor eye(int n) { return Tensor::eye(n); }

    // Global wrappers for UVNet
    inline Tensor leaky_relu(Tensor x, float negative_slope = 0.01) {
        Tensor out = x.clone();
        if (out.dtype_ == kFloat32) {
            for(auto& v : out.storage_->dataf_) {
                if (v < 0) v *= negative_slope;
            }
        }
        return out;
    }

    inline Tensor adaptive_avg_pool2d(Tensor x, std::initializer_list<int> output_size) {
        // Assume output_size is {1, 1} for global average pooling
        // x: [N, C, H, W] -> [N, C, 1, 1]
        if (x.dim() != 4) return x;
        int64_t N = x.size(0);
        int64_t C = x.size(1);
        int64_t H = x.size(2);
        int64_t W = x.size(3);
        
        Tensor out({N, C, 1, 1}, x.dtype_);
        
        for(int64_t n=0; n<N; ++n) {
            for(int64_t c=0; c<C; ++c) {
                float sum = 0;
                for(int64_t h=0; h<H; ++h) {
                    for(int64_t w=0; w<W; ++w) {
                        sum += x.at({n, c, h, w});
                    }
                }
                out.at({n, c, 0, 0}) = sum / (H * W);
            }
        }
        return out;
    }

    inline Tensor adaptive_avg_pool1d(Tensor x, std::initializer_list<int> output_size) {
        // Assume output_size is {1} for global average pooling
        // x: [N, C, L] -> [N, C, 1]
        if (x.dim() != 3) return x;
        int64_t N = x.size(0);
        int64_t C = x.size(1);
        int64_t L = x.size(2);
        
        Tensor out({N, C, 1}, x.dtype_);
        
        for(int64_t n=0; n<N; ++n) {
            for(int64_t c=0; c<C; ++c) {
                float sum = 0;
                for(int64_t l=0; l<L; ++l) {
                    sum += x.at({n, c, l});
                }
                out.at({n, c, 0}) = sum / L;
            }
        }
        return out;
    }

    inline Tensor conv2d(Tensor input, Tensor weight, Tensor bias = Tensor(), 
                         std::vector<int> stride = {1, 1}, std::vector<int> padding = {0, 0}, 
                         std::vector<int> dilation = {1, 1}, int groups = 1) {
        // input: [N, Cin, H, W]
        // weight: [Cout, Cin/groups, kH, kW]
        // bias: [Cout]
        
        int64_t N = input.size(0);
        int64_t Cin = input.size(1);
        int64_t H = input.size(2);
        int64_t W = input.size(3);
        
        int64_t Cout = weight.size(0);
        int64_t kH = weight.size(2);
        int64_t kW = weight.size(3);
        
        int64_t sH = stride.size() > 0 ? stride[0] : 1;
        int64_t sW = stride.size() > 1 ? stride[1] : sH;
        
        int64_t pH = padding.size() > 0 ? padding[0] : 0;
        int64_t pW = padding.size() > 1 ? padding[1] : pH;
        
        int64_t dH = dilation.size() > 0 ? dilation[0] : 1;
        int64_t dW = dilation.size() > 1 ? dilation[1] : dH;
        
        int64_t H_out = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        int64_t W_out = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
        
        Tensor out({N, Cout, H_out, W_out}, kFloat32);
        
        const float* in_ptr = input.storage_->dataf_.data();
        const float* w_ptr = weight.storage_->dataf_.data();
        const float* b_ptr = bias.defined() ? bias.storage_->dataf_.data() : nullptr;
        float* out_ptr = out.storage_->dataf_.data();
        
        for(int64_t n=0; n<N; ++n) {
            for(int64_t cout=0; cout<Cout; ++cout) {
                for(int64_t h_out=0; h_out<H_out; ++h_out) {
                    for(int64_t w_out=0; w_out<W_out; ++w_out) {
                        
                        float sum = 0;
                        if (b_ptr) sum = b_ptr[cout];
                        
                        for(int64_t cin=0; cin<Cin; ++cin) {
                            for(int64_t kh=0; kh<kH; ++kh) {
                                for(int64_t kw=0; kw<kW; ++kw) {
                                    int64_t h_in = h_out * sH - pH + kh * dH;
                                    int64_t w_in = w_out * sW - pW + kw * dW;
                                    
                                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                        float val = in_ptr[n*Cin*H*W + cin*H*W + h_in*W + w_in];
                                        float w_val = w_ptr[cout*Cin*kH*kW + cin*kH*kW + kh*kW + kw];
                                        sum += val * w_val;
                                    }
                                }
                            }
                        }
                        out_ptr[n*Cout*H_out*W_out + cout*H_out*W_out + h_out*W_out + w_out] = sum;
                    }
                }
            }
        }
        return out;
    }

    inline Tensor conv1d(Tensor input, Tensor weight, Tensor bias = Tensor(), 
                         std::vector<int> stride = {1}, std::vector<int> padding = {0}, 
                         std::vector<int> dilation = {1}, int groups = 1) {
        // input: [N, Cin, L]
        // weight: [Cout, Cin/groups, kL]
        
        int64_t N = input.size(0);
        int64_t Cin = input.size(1);
        int64_t L = input.size(2);
        
        int64_t Cout = weight.size(0);
        int64_t kL = weight.size(2);
        
        int64_t sL = stride.size() > 0 ? stride[0] : 1;
        int64_t pL = padding.size() > 0 ? padding[0] : 0;
        int64_t dL = dilation.size() > 0 ? dilation[0] : 1;
        
        int64_t L_out = (L + 2 * pL - dL * (kL - 1) - 1) / sL + 1;
        
        Tensor out({N, Cout, L_out}, kFloat32);
        
        const float* in_ptr = input.storage_->dataf_.data();
        const float* w_ptr = weight.storage_->dataf_.data();
        const float* b_ptr = bias.defined() ? bias.storage_->dataf_.data() : nullptr;
        float* out_ptr = out.storage_->dataf_.data();
        
        for(int64_t n=0; n<N; ++n) {
            for(int64_t cout=0; cout<Cout; ++cout) {
                for(int64_t l_out=0; l_out<L_out; ++l_out) {
                    
                    float sum = 0;
                    if (b_ptr) sum = b_ptr[cout];
                    
                    for(int64_t cin=0; cin<Cin; ++cin) {
                        for(int64_t kl=0; kl<kL; ++kl) {
                            int64_t l_in = l_out * sL - pL + kl * dL;
                            
                            if (l_in >= 0 && l_in < L) {
                                float val = in_ptr[n*Cin*L + cin*L + l_in];
                                float w_val = w_ptr[cout*Cin*kL + cin*kL + kl];
                                sum += val * w_val;
                            }
                        }
                    }
                    out_ptr[n*Cout*L_out + cout*L_out + l_out] = sum;
                }
            }
        }
        return out;
    }

    inline Tensor batch_norm(Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, 
                             bool training, double momentum, double eps, bool cudnn_enabled) {
        // input: [N, C, ...]
        // weight, bias, running_mean, running_var: [C]
        
        if (input.dim() < 2) return input;
        int64_t N = input.size(0);
        int64_t C = input.size(1);
        
        Tensor out = input.clone();
        int64_t spatial_size = input.numel() / (N * C);
        
        const float* rm = running_mean.storage_->dataf_.data();
        const float* rv = running_var.storage_->dataf_.data();
        const float* w = weight.defined() ? weight.storage_->dataf_.data() : nullptr;
        const float* b = bias.defined() ? bias.storage_->dataf_.data() : nullptr;
        float* out_data = out.storage_->dataf_.data();
        
        for(int64_t c=0; c<C; ++c) {
            float mean = rm[c];
            float var = rv[c];
            float inv_std = 1.0f / std::sqrt(var + (float)eps);
            float gamma = w ? w[c] : 1.0f;
            float beta = b ? b[c] : 0.0f;
            
            // Precompute scale and shift
            float scale = gamma * inv_std;
            float shift = beta - mean * scale;
            
            for(int64_t n=0; n<N; ++n) {
                for(int64_t s=0; s<spatial_size; ++s) {
                    int64_t idx = n * C * spatial_size + c * spatial_size + s;
                    out_data[idx] = out_data[idx] * scale + shift;
                }
            }
        }
        return out;
    }
    
    inline Tensor linear(Tensor input, Tensor weight, Tensor bias = Tensor()) {
        // input: [N, *, in_features]
        // weight: [out_features, in_features]
        // bias: [out_features]
        
        // Flatten input to [M, in_features]
        int64_t in_features = weight.size(1);
        int64_t out_features = weight.size(0);
        
        // Check input last dim
        if (input.sizes_.back() != in_features) throw std::runtime_error("linear: input shape mismatch");
        
        // Reshape input to 2D: [Batch, In]
        int64_t M = input.numel() / in_features;
        Tensor x_flat = input.view({M, in_features});
        
        // Matmul: [M, In] * [Out, In]^T -> [M, Out]
        // Transposing weight [Out, In] -> [In, Out]
        
        Tensor w_t({in_features, out_features}, weight.dtype_);
        for(int64_t i=0; i<out_features; ++i) {
            for(int64_t j=0; j<in_features; ++j) {
                if (weight.dtype_ == kFloat32) w_t.at({j, i}) = weight.at({i, j});
            }
        }
        
        Tensor out = matmul(x_flat, w_t);
        
        // Add bias
        if (bias.defined()) {
            for(int64_t i=0; i<M; ++i) {
                for(int64_t j=0; j<out_features; ++j) {
                    out.at({i, j}) += bias.at({j});
                }
            }
        }
        
        // Reshape back
        std::vector<int64_t> out_shape = input.sizes_;
        out_shape.back() = out_features;
        return out.view(out_shape);
    }

    // --------------------------------------------------------------------------
    // NN Namespace
    // --------------------------------------------------------------------------
    namespace nn {
        struct Module {
            std::map<std::string, std::shared_ptr<Module>> modules_;
            std::map<std::string, Tensor> params_;
            std::map<std::string, Tensor> buffers_;
            virtual ~Module() {}
            virtual Tensor forward(Tensor x) { return x; }

            // 1. shared_ptr overload
            template <typename T> 
            std::shared_ptr<T> register_module(std::string n, std::shared_ptr<T> m) { 
                modules_[n] = m; 
                return m; 
            }

            // 2. Module subclass overload (creates shared_ptr)
            template <typename T, typename std::enable_if<std::is_base_of<Module, T>::value, int>::type = 0>
            std::shared_ptr<T> register_module(std::string n, T m) { 
                auto p = std::make_shared<T>(m); 
                modules_[n] = p; 
                return p; 
            }

            // 3. Wrapper overload (e.g. TORCH_MODULE types that inherit from shared_ptr)
            template <typename T, typename std::enable_if<!std::is_base_of<Module, T>::value, int>::type = 0>
            T register_module(std::string n, T m) {
                // m should be convertible to shared_ptr<Module>
                modules_[n] = m; 
                return m;
            }

            void register_parameter(std::string n, Tensor t) { params_[n] = t; }

            std::vector<std::shared_ptr<Module>> children() const {
                std::vector<std::shared_ptr<Module>> v; for (auto& kv : modules_) v.push_back(kv.second); return v;
            }
            // Recursive parameter collection
            void _get_parameters(std::string prefix, std::map<std::string, Tensor*>& out) {
                for (auto& kv : params_) {
                    std::string name = prefix.empty() ? kv.first : prefix + "." + kv.first;
                    out[name] = &kv.second;
                }
                for (auto& kv : modules_) {
                    std::string sub_prefix = prefix.empty() ? kv.first : prefix + "." + kv.first;
                    kv.second->_get_parameters(sub_prefix, out);
                }
            }
            
            void _get_buffers(std::string prefix, std::map<std::string, Tensor*>& out) {
                for (auto& kv : buffers_) {
                    std::string name = prefix.empty() ? kv.first : prefix + "." + kv.first;
                    out[name] = &kv.second;
                }
                for (auto& kv : modules_) {
                    std::string sub_prefix = prefix.empty() ? kv.first : prefix + "." + kv.first;
                    kv.second->_get_buffers(sub_prefix, out);
                }
            }

            std::map<std::string, Tensor*> named_parameters() { 
                std::map<std::string, Tensor*> out;
                _get_parameters("", out);
                return out;
            }
            
            std::map<std::string, Tensor*> named_buffers() { 
                std::map<std::string, Tensor*> out;
                _get_buffers("", out);
                return out;
            }

        };

        struct LinearOptions { int in, out; bool b; LinearOptions(int i, int o) :in(i), out(o), b(true) {} LinearOptions& bias(bool v) { b = v; return *this; } };
        struct LinearImpl : Module { 
            Tensor weight, bias;
            LinearImpl(LinearOptions o) {
                weight = Tensor::zeros({(int64_t)o.out, (int64_t)o.in}, kFloat32);
                // Simple initialization
                weight.fill(0.01f); 
                register_parameter("weight", weight);
                if (o.b) {
                    bias = Tensor::zeros({(int64_t)o.out}, kFloat32);
                    register_parameter("bias", bias);
                }
            } 
            Tensor forward(Tensor x) override { 
                return breptorch::linear(x, weight, bias); 
            } 
        };
        
        struct ReLUImpl : Module { 
            Tensor forward(Tensor x) override { 
                // Simple ReLU: max(0, x)
                Tensor out = x.clone();
                if (out.dtype_ == kFloat32) {
                    for(auto& v : out.storage_->dataf_) if(v < 0) v = 0;
                }
                return out;
            } 
        };
        
        struct SequentialImpl : Module {
            std::vector<std::shared_ptr<Module>> ordered_modules;
            
            void push_back(std::string n, std::shared_ptr<Module> m) { 
                register_module(n, m); 
                ordered_modules.push_back(m);
            }
            // Add shared_ptr overload
            void push_back(std::shared_ptr<Module> m) { 
                register_module("layer_" + std::to_string(ordered_modules.size()), m); 
                ordered_modules.push_back(m);
            }
            
            Tensor forward(Tensor x) override { 
                for(auto& m : ordered_modules) {
                    x = m->forward(x);
                }
                return x; 
            }
        };
        
        struct BatchNorm1dOptions { int f; double e; BatchNorm1dOptions(int i) :f(i), e(1e-5) {} BatchNorm1dOptions& eps(double v) { e = v; return *this; } };
        struct BatchNorm1dImpl : Module { 
            Tensor weight, bias, running_mean, running_var;
            double eps;
            
            BatchNorm1dImpl(BatchNorm1dOptions o) : eps(o.e) {
                weight = Tensor::ones({(int64_t)o.f}, kFloat32);
                bias = Tensor::zeros({(int64_t)o.f}, kFloat32);
                running_mean = Tensor::zeros({(int64_t)o.f}, kFloat32);
                running_var = Tensor::ones({(int64_t)o.f}, kFloat32);
                
                register_parameter("weight", weight);
                register_parameter("bias", bias);
                // Buffers are not parameters (no gradient), but state
                buffers_["running_mean"] = running_mean;
                buffers_["running_var"] = running_var;
            } 
            
            Tensor forward(Tensor x) override { 
                // Update internal references in case they were modified externally (e.g. load_state_dict)
                if (params_.count("weight")) weight = params_["weight"];
                if (params_.count("bias")) bias = params_["bias"];
                if (buffers_.count("running_mean")) running_mean = buffers_["running_mean"];
                if (buffers_.count("running_var")) running_var = buffers_["running_var"];
                
                return breptorch::batch_norm(x, weight, bias, running_mean, running_var, false, 0.1, eps, true);
            } 
        };

        // Factory
        inline std::shared_ptr<LinearImpl> Linear(LinearOptions o) { return std::make_shared<LinearImpl>(o); }
        inline std::shared_ptr<ReLUImpl> ReLU() { return std::make_shared<ReLUImpl>(); }
        inline std::shared_ptr<SequentialImpl> Sequential() { return std::make_shared<SequentialImpl>(); }
        inline std::shared_ptr<BatchNorm1dImpl> BatchNorm1d(BatchNorm1dOptions o) { return std::make_shared<BatchNorm1dImpl>(o); }

        // Aliases
        using LinearPtr = std::shared_ptr<LinearImpl>;
        using ReLUPtr = std::shared_ptr<ReLUImpl>;
        using SequentialPtr = std::shared_ptr<SequentialImpl>;
        using BatchNorm1dPtr = std::shared_ptr<BatchNorm1dImpl>;

        namespace functional {
            struct Conv2dFuncOptions { int s = 1, p = 0; Conv2dFuncOptions& stride(int v) { s = v; return *this; } Conv2dFuncOptions& padding(int v) { p = v; return *this; } };
            struct Conv1dFuncOptions { int s = 1, p = 0; Conv1dFuncOptions& stride(int v) { s = v; return *this; } Conv1dFuncOptions& padding(int v) { p = v; return *this; } };
            struct BatchNormFuncOptions { Tensor w, b; double e; double m = 0.1; BatchNormFuncOptions& weight(Tensor v) { w = v; return *this; } BatchNormFuncOptions& bias(Tensor v) { b = v; return *this; } BatchNormFuncOptions& training(bool) { return *this; } BatchNormFuncOptions& eps(double v) { e = v; return *this; } BatchNormFuncOptions& momentum(double v) { m = v; return *this; } };

            inline Tensor conv2d(Tensor x, Tensor w, const Conv2dFuncOptions& o) { 
                return breptorch::conv2d(x, w, Tensor(), 
                                         {o.s, o.s}, {o.p, o.p});
            }
            inline Tensor conv1d(Tensor x, Tensor w, const Conv1dFuncOptions& o) { 
                return breptorch::conv1d(x, w, Tensor(), 
                                         {o.s}, {o.p});
            }
            inline Tensor linear(Tensor x, Tensor w, Tensor b) { 
                return breptorch::linear(x, w, b);
            }
            inline Tensor batch_norm(Tensor x, Tensor m, Tensor v, const BatchNormFuncOptions& o) { 
                return breptorch::batch_norm(x, o.w, o.b, m, v, false, o.m, o.e, true);
            }
            inline Tensor leaky_relu(Tensor x, float s) { 
                return breptorch::leaky_relu(x, s);
            }
            inline Tensor adaptive_avg_pool2d(Tensor x, std::initializer_list<int> s) { 
                return breptorch::adaptive_avg_pool2d(x, s);
            }
            inline Tensor adaptive_avg_pool1d(Tensor x, std::initializer_list<int> s) { 
                return breptorch::adaptive_avg_pool1d(x, s);
            }
        }
    } // namespace nn


#define TORCH_MODULE(Name) \
    struct Name : public std::shared_ptr<Name##Impl> { \
        using std::shared_ptr<Name##Impl>::shared_ptr; \
        Name(Name##Impl* ptr) : std::shared_ptr<Name##Impl>(ptr) {} \
        Name() : std::shared_ptr<Name##Impl>(nullptr) {} \
        template<typename... Args> \
        Name(Args&&... args) : std::shared_ptr<Name##Impl>(std::make_shared<Name##Impl>(std::forward<Args>(args)...)) {} \
    };

    // Global operator overloads
    inline Tensor operator*(float v, const Tensor& t) { return t * v; }

} // namespace breptorch