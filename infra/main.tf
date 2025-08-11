# Existing Resource Group
data "azurerm_resource_group" "rg" {
  name = var.resource_group_name
}

# Existing App Service Plan
data "azurerm_service_plan" "plan" {
  name                = var.app_service_plan_name
  resource_group_name = data.azurerm_resource_group.rg.name
}

# Create the Linux Web App, using existing RG & Plan
resource "azurerm_linux_web_app" "app" {
  name                = var.webapp_name
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  service_plan_id     = data.azurerm_service_plan.plan.id

  site_config {
    application_stack {
      docker_image     = "nginx"
      docker_image_tag = "latest"
    }
    always_on = true
  }

  https_only = true
}
