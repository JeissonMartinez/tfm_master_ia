// =============================================================================
// dashboard.h — Dashboard HTML embebido servido desde flash
// =============================================================================
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Registrar handlers HTTP para el dashboard.
 *        Debe llamarse después de network_init().
 *        Sirve index.html en GET /
 */
void dashboard_register_handlers(void);

#ifdef __cplusplus
}
#endif
