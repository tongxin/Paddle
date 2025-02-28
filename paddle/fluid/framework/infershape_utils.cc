/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"

#include <algorithm>
#include <string>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/compat/arg_map_context.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/compat/op_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace framework {

class InferShapeArgumentMappingContext : public phi::ArgumentMappingContext {
 public:
  explicit InferShapeArgumentMappingContext(const InferShapeContext& ctx)
      : ctx_(ctx) {}

  bool HasInput(const std::string& name) const override {
    return ctx_.HasInput(name);
  }

  bool HasOutput(const std::string& name) const override {
    return ctx_.HasOutput(name);
  }

  bool HasAttr(const std::string& name) const override {
    return ctx_.HasAttr(name);
  }

  paddle::any Attr(const std::string& name) const override {
    auto& attr = ctx_.Attrs().GetAttr(name);
    return GetAttrValue(attr);
  }

  size_t InputSize(const std::string& name) const override {
    if (ctx_.HasInputs(name)) {
      return ctx_.Inputs(name).size();
    } else if (ctx_.HasInput(name)) {
      return 1;
    }
    return 0;
  }

  size_t OutputSize(const std::string& name) const override {
    return ctx_.Outputs(name).size();
  }

  bool IsDenseTensorInput(const std::string& name) const override {
    auto var_type = ctx_.GetInputVarType(name);
    return var_type == proto::VarType::LOD_TENSOR;
  }

  bool IsDenseTensorInputs(const std::string& name) const override {
    auto var_types = ctx_.GetInputsVarType(name);
    return std::all_of(var_types.begin(), var_types.end(),
                       [](const proto::VarType::Type& type) {
                         return type == proto::VarType::LOD_TENSOR;
                       });
  }

  bool IsSelectedRowsInput(const std::string& name) const override {
    auto var_type = ctx_.GetInputVarType(name);
    return var_type == proto::VarType::SELECTED_ROWS;
  }

  bool IsDenseTensorVectorInput(const std::string& name) const override {
    auto var_types = ctx_.GetInputsVarType(name);
    return std::all_of(var_types.begin(), var_types.end(),
                       [](const proto::VarType::Type& type) {
                         return type == proto::VarType::LOD_TENSOR_ARRAY;
                       });
  }

  bool IsDenseTensorOutput(const std::string& name) const override {
    auto var_types = ctx_.GetOutputsVarType(name);
    return std::all_of(var_types.begin(), var_types.end(),
                       [](const proto::VarType::Type& type) {
                         return type == proto::VarType::LOD_TENSOR;
                       });
  }

  bool IsSelectedRowsOutput(const std::string& name) const override {
    auto var_types = ctx_.GetOutputsVarType(name);
    return std::all_of(var_types.begin(), var_types.end(),
                       [](const proto::VarType::Type& type) {
                         return type == proto::VarType::SELECTED_ROWS;
                       });
  }

  bool IsForInferShape() const override { return true; }

  bool IsRuntime() const override { return ctx_.IsRuntime(); }

 private:
  const InferShapeContext& ctx_;
};

int64_t CompatMetaTensor::numel() const {
  if (is_runtime_) {
    auto* var = BOOST_GET_CONST(Variable*, var_);
    return var->Get<Tensor>().numel();
  } else {
    auto* var = BOOST_GET_CONST(VarDesc*, var_);
    return var->ElementSize();
  }
}

DDim CompatMetaTensor::dims() const {
  if (is_runtime_) {
    auto* var = BOOST_GET_CONST(Variable*, var_);
    if (var->IsType<phi::DenseTensor>()) {
      return var->Get<phi::DenseTensor>().dims();
    } else if (var->IsType<phi::SelectedRows>()) {
      return var->Get<phi::SelectedRows>().dims();
    } else if (var->IsType<framework::LoDTensorArray>()) {
      // use tensor array size as dims
      auto& tensor_array = var->Get<framework::LoDTensorArray>();
      return phi::make_ddim({static_cast<int64_t>(tensor_array.size())});
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently, only can get dims from DenseTensor or SelectedRows or "
          "DenseTensorArray."));
    }
  } else {
    auto* var = BOOST_GET_CONST(VarDesc*, var_);

    return var->GetShape().empty() ? phi::make_ddim({0UL})
                                   : phi::make_ddim(var->GetShape());
  }
}

phi::DataType CompatMetaTensor::dtype() const {
  if (is_runtime_) {
    auto* var = BOOST_GET_CONST(Variable*, var_);
    if (var->IsType<phi::DenseTensor>()) {
      return var->Get<phi::DenseTensor>().dtype();
    } else if (var->IsType<phi::SelectedRows>()) {
      return var->Get<phi::SelectedRows>().dtype();
    } else if (var->IsType<framework::LoDTensorArray>()) {
      // NOTE(chenweihang): do nothing
      // Unsupported get dtype from LoDTensorArray now
      return phi::DataType::UNDEFINED;
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently, only can get dtype from DenseTensor or SelectedRows."));
    }
  } else {
    auto* var = BOOST_GET_CONST(VarDesc*, var_);
    return paddle::framework::TransToPhiDataType(var->GetDataType());
  }
}

DataLayout CompatMetaTensor::layout() const {
  if (is_runtime_) {
    auto* var = BOOST_GET_CONST(Variable*, var_);
    if (var->IsType<phi::DenseTensor>()) {
      return var->Get<phi::DenseTensor>().layout();
    } else if (var->IsType<phi::SelectedRows>()) {
      return var->Get<phi::SelectedRows>().layout();
    } else if (var->IsType<framework::LoDTensorArray>()) {
      // NOTE(chenweihang): do nothing
      // Unsupported get layout from LoDTensorArray now
      return phi::DataLayout::UNDEFINED;
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently, only can get layout from DenseTensor or "
          "SelectedRows."));
    }
  } else {
    // NOTE(chenweihang): do nothing
    // Unsupported get layout for VarDesc now
    return DataLayout::UNDEFINED;
  }
}

void CompatMetaTensor::set_dims(const DDim& dims) {
  if (is_runtime_) {
    auto* var = BOOST_GET(Variable*, var_);
    if (var->IsType<phi::DenseTensor>()) {
      auto* tensor = var->GetMutable<phi::DenseTensor>();
      phi::DenseTensorUtils::GetMutableMeta(tensor)->dims = dims;
    } else if (var->IsType<phi::SelectedRows>()) {
      auto* tensor = var->GetMutable<phi::SelectedRows>()->mutable_value();
      phi::DenseTensorUtils::GetMutableMeta(tensor)->dims = dims;
    } else if (var->IsType<framework::LoDTensorArray>()) {
      auto* tensor_array = var->GetMutable<framework::LoDTensorArray>();
      // Note: Here I want enforce `tensor_array->size() == 0UL`, because
      // inplace using on LoDTensorArray is dangerous, but the unittest
      // `test_list` contains this behavior
      PADDLE_ENFORCE_EQ(dims.size(), 1UL,
                        platform::errors::InvalidArgument(
                            "LoDTensorArray can only have one dimension."));
      // only set the array size for LoDTensorArray input
      tensor_array->resize(dims[0]);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently, only can set dims from DenseTensor or SelectedRows."));
    }
  } else {
    auto* var = BOOST_GET(VarDesc*, var_);
    var->SetShape(vectorize(dims));
  }
}

void CompatMetaTensor::set_dtype(phi::DataType dtype) {
  if (is_runtime_) {
    auto* var = BOOST_GET(Variable*, var_);
    if (var->IsType<phi::DenseTensor>()) {
      auto* tensor = var->GetMutable<phi::DenseTensor>();
      phi::DenseTensorUtils::GetMutableMeta(tensor)->dtype = dtype;
    } else if (var->IsType<phi::SelectedRows>()) {
      auto* tensor = var->GetMutable<phi::SelectedRows>()->mutable_value();
      phi::DenseTensorUtils::GetMutableMeta(tensor)->dtype = dtype;
    } else if (var->IsType<framework::LoDTensorArray>()) {
      // NOTE(chenweihang): do nothing
      // Unsupported set dtype for LoDTensorArray now
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently, only can set dtype from DenseTensor or SelectedRows."));
    }
  } else {
    auto* var = BOOST_GET(VarDesc*, var_);
    var->SetDataType(paddle::framework::TransToProtoVarType(dtype));
  }
}

void CompatMetaTensor::set_layout(DataLayout layout) {
  if (is_runtime_) {
    auto* var = BOOST_GET(Variable*, var_);
    if (var->IsType<phi::DenseTensor>()) {
      auto* tensor = var->GetMutable<phi::DenseTensor>();
      phi::DenseTensorUtils::GetMutableMeta(tensor)->layout = layout;
    } else if (var->IsType<phi::SelectedRows>()) {
      auto* tensor = var->GetMutable<phi::SelectedRows>()->mutable_value();
      phi::DenseTensorUtils::GetMutableMeta(tensor)->layout = layout;
    } else if (var->IsType<framework::LoDTensorArray>()) {
      // NOTE(chenweihang): do nothing
      // Unsupported set dtype for LoDTensorArray now
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently, only can set layout from DenseTensor or "
          "SelectedRows."));
    }
  } else {
    // NOTE(chenweihang): do nothing
    // Unsupported set layout for VarDesc now
  }
}

void CompatMetaTensor::share_lod(const MetaTensor& meta_tensor) {
  if (is_runtime_) {
    auto* var = BOOST_GET(Variable*, var_);
    if (var->IsType<phi::DenseTensor>()) {
      auto* tensor = var->GetMutable<phi::DenseTensor>();
      phi::DenseTensorUtils::GetMutableMeta(tensor)->lod =
          static_cast<const CompatMetaTensor&>(meta_tensor).GetRuntimeLoD();
    } else {
      // NOTE(chenweihang): do nothing
      // only LoDTensor need to share lod
    }
  } else {
    auto* var = BOOST_GET(VarDesc*, var_);
    var->SetLoDLevel(
        static_cast<const CompatMetaTensor&>(meta_tensor).GetCompileTimeLoD());
  }
}

void CompatMetaTensor::share_dims(const MetaTensor& meta_tensor) {
  set_dims(meta_tensor.dims());
  if (is_runtime_) {
    auto* var = BOOST_GET(Variable*, var_);
    if (var->IsType<phi::SelectedRows>()) {
      auto* selected_rows = var->GetMutable<phi::SelectedRows>();
      auto& input_selected_rows =
          static_cast<const CompatMetaTensor&>(meta_tensor).GetSelectedRows();
      selected_rows->set_rows(input_selected_rows.rows());
      selected_rows->set_height(input_selected_rows.height());
    }
  }
}

void CompatMetaTensor::share_meta(const MetaTensor& meta_tensor) {
  share_dims(meta_tensor);
  set_dtype(meta_tensor.dtype());
  set_layout(meta_tensor.layout());
  // special case: share lod of LoDTensor
  share_lod(meta_tensor);
}

void CompatInferMetaContext::EmplaceBackInput(CompatMetaTensor input) {
  int index = compat_inputs_.size();
  compat_inputs_.emplace_back(std::move(input));
  input_range_.emplace_back(std::pair<int, int>(index, index + 1));
}
void CompatInferMetaContext::EmplaceBackOutput(CompatMetaTensor output) {
  int index = compat_outputs_.size();
  compat_outputs_.emplace_back(std::move(output));
  output_range_.emplace_back(std::pair<int, int>(index, index + 1));
}

void CompatInferMetaContext::EmplaceBackInputs(
    paddle::SmallVector<CompatMetaTensor, phi::kInputSmallVectorSize> inputs) {
  int index = compat_inputs_.size();
  input_range_.emplace_back(std::pair<int, int>(index, index + inputs.size()));
  compat_inputs_.insert(compat_inputs_.end(),
                        std::make_move_iterator(inputs.begin()),
                        std::make_move_iterator(inputs.end()));
}

void CompatInferMetaContext::EmplaceBackOutputs(
    paddle::SmallVector<CompatMetaTensor, phi::kOutputSmallVectorSize>
        outputs) {
  int index = compat_outputs_.size();
  output_range_.emplace_back(
      std::pair<int, int>(index, index + outputs.size()));
  compat_outputs_.insert(compat_outputs_.end(),
                         std::make_move_iterator(outputs.begin()),
                         std::make_move_iterator(outputs.end()));
}

const phi::MetaTensor& CompatInferMetaContext::InputAt(size_t idx) const {
  return compat_inputs_.at(idx);
}

paddle::optional<const phi::MetaTensor&>
CompatInferMetaContext::OptionalInputAt(size_t idx) const {
  const auto& input = compat_inputs_.at(idx);
  return input.initialized()
             ? paddle::optional<const phi::MetaTensor&>{input}
             : paddle::optional<const phi::MetaTensor&>{paddle::none};
}

std::vector<const phi::MetaTensor*> CompatInferMetaContext::InputsBetween(
    size_t start, size_t end) const {
  std::vector<const phi::MetaTensor*> result;
  result.reserve(end - start);

  for (size_t i = start; i < end; ++i) {
    auto& in = compat_inputs_.at(i);
    result.emplace_back(in.initialized() ? &in : nullptr);
  }

  return result;
}

paddle::optional<const std::vector<const phi::MetaTensor*>>
CompatInferMetaContext::OptionalInputsBetween(size_t start, size_t end) const {
  const auto& first = compat_inputs_.at(start);

  if (first.initialized()) {
    std::vector<const phi::MetaTensor*> result;
    result.reserve(end - start);

    for (size_t i = start; i < end; ++i) {
      auto& in = compat_inputs_.at(i);
      result.emplace_back(in.initialized() ? &in : nullptr);
    }

    return paddle::optional<const std::vector<const phi::MetaTensor*>>(result);
  }
  return paddle::optional<const std::vector<const phi::MetaTensor*>>(
      paddle::none);
}

phi::MetaTensor* CompatInferMetaContext::MutableOutputAt(size_t idx) {
  auto& out = compat_outputs_.at(idx);
  return out.initialized() ? &out : nullptr;
}

std::vector<phi::MetaTensor*> CompatInferMetaContext::MutableOutputBetween(
    size_t start, size_t end) {
  std::vector<phi::MetaTensor*> result;
  result.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    auto& out = compat_outputs_.at(i);
    result.emplace_back(out.initialized() ? &out : nullptr);
  }
  return result;
}

CompatInferMetaContext BuildInferMetaContext(InferShapeContext* ctx,
                                             const std::string& op_type) {
  // 1. get kernel args
  auto* arg_map_fn = ctx->GetPhiArgumentMappingFn();
  InferShapeArgumentMappingContext arg_map_context(*ctx);
  phi::KernelSignature signature = arg_map_fn
                                       ? (*arg_map_fn)(arg_map_context)
                                       : *ctx->GetPhiDefaultKernelSignature();
  VLOG(3) << "BuildInferMetaContext: op kernel signature - " << signature;

  // 2. build infermeta context
  CompatInferMetaContext infer_meta_context(
      {ctx->IsRuntime(), ctx->IsRunMKLDNNKernel()});

  const auto& input_names = signature.input_names;
  const auto& attr_names = signature.attr_names;
  const auto& output_names = signature.output_names;

  const auto& args_def =
      phi::KernelFactory::Instance().GetFirstKernelArgsDef(signature.name);
  const auto& attr_defs = args_def.attribute_defs();

  for (auto& in_name : input_names) {
    if (ctx->HasInputs(in_name)) {
      auto input_var = std::move(ctx->GetInputVarPtrs(in_name));
      if (input_var.size() == 1) {
        infer_meta_context.EmplaceBackInput(
            std::move(CompatMetaTensor(input_var[0], ctx->IsRuntime())));
      } else {
        paddle::SmallVector<CompatMetaTensor, phi::kInputSmallVectorSize>
            inputs;
        for (const auto& in : input_var) {
          inputs.emplace_back(
              std::move(CompatMetaTensor(in, ctx->IsRuntime())));
        }
        infer_meta_context.EmplaceBackInputs(std::move(inputs));
      }
    } else {
      infer_meta_context.EmplaceBackInput(
          std::move(CompatMetaTensor(ctx->IsRuntime())));
    }
  }

  VLOG(6) << "BuildInferMetaContext: Done inputs";

  auto attr_reader = ctx->Attrs();
  for (size_t i = 0; i < attr_names.size(); ++i) {
    auto& attr_name = attr_names[i];
    if (attr_defs[i].type_index == phi::AttributeType::INT_ARRAY) {
      // When attr is a vector_tensor or tensor, transform it to IntArray
      if (ctx->HasInputs(attr_name) || ctx->HasInput(attr_name)) {
        auto infershape_inputs = std::move(ctx->GetInputVarPtrs(attr_name));
        if (ctx->IsRuntime()) {
          // If is in runtime, we will get tensor's value for IntArray
          // and push it into attrs
          std::vector<Variable*> vars;
          vars.reserve(infershape_inputs.size());
          for (size_t i = 0; i < infershape_inputs.size(); i++) {
            vars.push_back(BOOST_GET_CONST(Variable*, infershape_inputs[i]));
          }
          if (infershape_inputs.size() != 1) {
            infer_meta_context.EmplaceBackAttr(
                std::move(experimental::MakePhiIntArrayFromVarList(vars)));
          } else {
            infer_meta_context.EmplaceBackAttr(
                std::move(experimental::MakePhiIntArrayFromVar(*vars[0])));
          }
        } else {
          // If is not in runtime, we will set default value(-1) for IntArray
          std::vector<VarDesc*> vars;
          vars.reserve(infershape_inputs.size());
          for (size_t i = 0; i < infershape_inputs.size(); ++i) {
            vars.push_back(BOOST_GET_CONST(VarDesc*, infershape_inputs[i]));
          }

          int64_t num_ele = 0;
          if (vars.size() == 1) {
            num_ele = 1;
            const auto& tensor_dims = vars[0]->GetShape();
            for (size_t i = 0; i < tensor_dims.size(); ++i) {
              num_ele *= tensor_dims[i];
            }

            if (num_ele <= 0) {
              PADDLE_THROW(platform::errors::Unimplemented(
                  "Invalid number for construct phi::IntArray, expected "
                  "number > 0, but actually is %d. ",
                  num_ele));
            }

          } else {
            num_ele = vars.size();
          }
          phi::IntArray tensor_attr(std::vector<int32_t>(num_ele, -1));
          tensor_attr.SetFromTensor(true);
          infer_meta_context.EmplaceBackAttr(std::move(tensor_attr));
        }
      } else if (ctx->HasAttr(attr_name)) {
        auto& attr = attr_reader.GetAttr(attr_name);
        if (std::type_index(attr.type()) ==
            std::type_index(typeid(std::vector<int32_t>))) {
          infer_meta_context.EmplaceBackAttr(std::move(
              phi::IntArray(BOOST_GET_CONST(std::vector<int32_t>, attr))));
        } else if (std::type_index(attr.type()) ==
                   std::type_index(typeid(std::vector<int64_t>))) {
          infer_meta_context.EmplaceBackAttr(std::move(
              phi::IntArray(BOOST_GET_CONST(std::vector<int64_t>, attr))));
        } else if (std::type_index(attr.type()) ==
                   std::type_index(typeid(int))) {
          infer_meta_context.EmplaceBackAttr(
              phi::IntArray({BOOST_GET_CONST(int, attr)}));
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Unsupported cast op attribute `%s` to IntArray when "
              "construct InferMetaContext.",
              attr_name));
        }
      }
    } else if (attr_defs[i].type_index == phi::AttributeType::SCALAR) {
      if (ctx->HasAttr(attr_name)) {
        // TODO(chentianyu03): support other attrs later
        auto& attr = attr_reader.GetAttr(attr_name);
        if (std::type_index(attr.type()) == std::type_index(typeid(float))) {
          infer_meta_context.EmplaceBackAttr(
              phi::Scalar(BOOST_GET_CONST(float, attr)));
        } else if (std::type_index(attr.type()) ==
                   std::type_index(typeid(std::string))) {
          infer_meta_context.EmplaceBackAttr(
              phi::Scalar(BOOST_GET_CONST(std::string, attr)));
        } else if (std::type_index(attr.type()) ==
                   std::type_index(typeid(int))) {
          infer_meta_context.EmplaceBackAttr(
              phi::Scalar(BOOST_GET_CONST(int, attr)));
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Unsupported cast op attribute `%s` to Scalar when construct "
              "InferMetaContext.",
              attr_name));
        }
      } else if (ctx->HasInput(attr_name)) {
        auto infershape_input = std::move(ctx->GetInputVarPtrs(attr_name));
        if (infershape_input.size() == 1) {
          if (ctx->IsRuntime()) {
            Variable* var = BOOST_GET_CONST(Variable*, infershape_input[0]);
            infer_meta_context.EmplaceBackAttr(
                std::move(experimental::MakePhiScalarFromVar(*var)));
          } else {
            phi::Scalar tensor_scalar(-1);
            tensor_scalar.SetFromTensor(true);
            infer_meta_context.EmplaceBackAttr(std::move(tensor_scalar));
          }
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Invalid input.size() when cast op attribute `%s` to Scalar, "
              "expected 1, but actually is %d .",
              attr_name, infershape_input.size()));
        }
      }
    } else if (attr_defs[i].type_index == phi::AttributeType::SCALARS) {
      auto& attr = attr_reader.GetAttr(attr_name);
      if (std::type_index(attr.type()) ==
          std::type_index(typeid(std::vector<int32_t>))) {
        const auto& vec = BOOST_GET_CONST(std::vector<int32_t>, attr);
        std::vector<phi::Scalar> scalar_list;
        scalar_list.reserve(vec.size());
        for (const auto& val : vec) {
          scalar_list.emplace_back(val);
        }
        infer_meta_context.EmplaceBackAttr(std::move(scalar_list));
      } else if (std::type_index(attr.type()) ==
                 std::type_index(typeid(std::vector<int64_t>))) {
        const auto& vec = BOOST_GET_CONST(std::vector<int64_t>, attr);
        std::vector<phi::Scalar> scalar_list;
        scalar_list.reserve(vec.size());
        for (const auto& val : vec) {
          scalar_list.emplace_back(val);
        }
        infer_meta_context.EmplaceBackAttr(std::move(scalar_list));
      } else if (std::type_index(attr.type()) ==
                 std::type_index(typeid(std::vector<float>))) {
        const auto& vec = BOOST_GET_CONST(std::vector<float>, attr);
        std::vector<phi::Scalar> scalar_list;
        scalar_list.reserve(vec.size());
        for (const auto& val : vec) {
          scalar_list.emplace_back(val);
        }
        infer_meta_context.EmplaceBackAttr(std::move(scalar_list));
      } else if (std::type_index(attr.type()) ==
                 std::type_index(typeid(std::vector<double>))) {
        const auto& vec = BOOST_GET_CONST(std::vector<double>, attr);
        std::vector<phi::Scalar> scalar_list;
        scalar_list.reserve(vec.size());
        for (const auto& val : vec) {
          scalar_list.emplace_back(val);
        }
        infer_meta_context.EmplaceBackAttr(std::move(scalar_list));
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported cast op attribute `%s` to vector<Scalar> when "
            "construct InferMetaContext.",
            attr_names[i]));
      }
    } else if (ctx->HasAttr(attr_name)) {
      // Emplace Back Attr according to the type of attr.
      auto& attr = attr_reader.GetAttr(attr_name);
      if (attr_defs[i].type_index == phi::AttributeType::BOOL) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(bool, attr));
      } else if (attr_defs[i].type_index == phi::AttributeType::INT32) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(int, attr));
      } else if (attr_defs[i].type_index == phi::AttributeType::INT64) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(int64_t, attr));
      } else if (attr_defs[i].type_index == phi::AttributeType::FLOAT32) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(float, attr));
      } else if (attr_defs[i].type_index == phi::AttributeType::STRING) {
        infer_meta_context.EmplaceBackAttr(BOOST_GET_CONST(std::string, attr));
      } else if (attr_defs[i].type_index == phi::AttributeType::BOOLS) {
        infer_meta_context.EmplaceBackAttr(
            BOOST_GET_CONST(std::vector<bool>, attr));
      } else if (attr_defs[i].type_index == phi::AttributeType::INT32S) {
        infer_meta_context.EmplaceBackAttr(
            BOOST_GET_CONST(std::vector<int>, attr));
      } else if (attr_defs[i].type_index == phi::AttributeType::INT64S) {
        if (std::type_index(attr.type()) ==
            std::type_index(typeid(std::vector<int>))) {
          // Emplace Back Attr according to the type of Phi_Kernel args.
          const auto& vector_int_attr = BOOST_GET_CONST(std::vector<int>, attr);
          const std::vector<int64_t> vector_int64_attr(vector_int_attr.begin(),
                                                       vector_int_attr.end());
          infer_meta_context.EmplaceBackAttr(vector_int64_attr);
        } else {
          infer_meta_context.EmplaceBackAttr(
              BOOST_GET_CONST(std::vector<int64_t>, attr));
        }
      } else if (attr_defs[i].type_index == phi::AttributeType::FLOAT32S) {
        infer_meta_context.EmplaceBackAttr(
            BOOST_GET_CONST(std::vector<float>, attr));
      } else if (attr_defs[i].type_index == phi::AttributeType::FLOAT64S) {
        infer_meta_context.EmplaceBackAttr(
            BOOST_GET_CONST(std::vector<double>, attr));
      } else if (attr_defs[i].type_index == phi::AttributeType::STRINGS) {
        infer_meta_context.EmplaceBackAttr(
            BOOST_GET_CONST(std::vector<std::string>, attr));
      } else if (attr_defs[i].type_index == phi::AttributeType::DATA_TYPE) {
        auto data_type = paddle::framework::TransToPhiDataType(
            static_cast<framework::proto::VarType::Type>(
                BOOST_GET_CONST(int, attr)));
        infer_meta_context.EmplaceBackAttr(data_type);
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported attribute type is received when call "
            "InferShapeFunctor."));
      }
    } else if (ctx->HasInput(attr_name)) {
      // convert from data
      if (attr_defs[i].type_index == phi::AttributeType::INT32) {
        if (ctx->IsRuntime()) {
          auto infershape_inputs = std::move(ctx->GetInputVarPtrs(attr_name));
          auto var_temp = BOOST_GET_CONST(Variable*, infershape_inputs[i]);
          auto val = experimental::MakePhiScalarFromVar(*var_temp);
          int32_t val_int = val.template to<int32_t>();
          infer_meta_context.EmplaceBackAttr(val_int);
        } else {
          infer_meta_context.EmplaceBackAttr(-1);
        }
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Get value from variable only support int yet"));
      }
    }
  }

  VLOG(6) << "BuildInferMetaContext: Done attrs";

  for (auto& out_name : output_names) {
    if (ctx->HasOutputs(out_name, true)) {
      auto output_var = std::move(ctx->GetOutputVarPtrs(out_name));
      if (output_var.size() == 1) {
        infer_meta_context.EmplaceBackOutput(
            std::move(CompatMetaTensor(output_var[0], ctx->IsRuntime())));
      } else {
        paddle::SmallVector<CompatMetaTensor, phi::kOutputSmallVectorSize>
            outputs;
        for (const auto& out : output_var) {
          if (ctx->IsRuntime()) {
            if (BOOST_GET_CONST(Variable*, out)) {
              outputs.emplace_back(
                  std::move(CompatMetaTensor(out, ctx->IsRuntime())));
              continue;
            }
          } else if (BOOST_GET_CONST(VarDesc*, out)) {
            outputs.emplace_back(
                std::move(CompatMetaTensor(out, ctx->IsRuntime())));
            continue;
          }
          outputs.emplace_back(std::move(CompatMetaTensor(ctx->IsRuntime())));
        }
        infer_meta_context.EmplaceBackOutputs(std::move(outputs));
      }
    } else {
      infer_meta_context.EmplaceBackOutput(
          std::move(CompatMetaTensor(ctx->IsRuntime())));
    }
  }

  VLOG(6) << "BuildInferMetaContext: Done outputs";

  return infer_meta_context;
}

}  // namespace framework
}  // namespace paddle
