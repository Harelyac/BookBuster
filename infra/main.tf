# Use existing Resource Group
data "azurerm_resource_group" "rg" {
  name = var.resource_group_name
}

# Create App Service Plan
resource "azurerm_service_plan" "plan" {
  name                = var.app_service_plan_name
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
  os_type             = "Linux"
  sku_name            = "B1" # Adjust to your needs (S1, P1v3, etc.)
}

# Create the Linux Web App
resource "azurerm_linux_web_app" "app" {
  name                = var.webapp_name
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  service_plan_id     = azurerm_service_plan.plan.id

  site_config {
    application_stack {
      docker_image     = "nginx"
      docker_image_tag = "latest"
    }
    always_on = true
  }

  https_only = true
}
