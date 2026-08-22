package com.sih.document_screening.config;

import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.web.cors.CorsConfigurationSource;

@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private final CorsConfigurationSource corsConfigurationSource;

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            .cors(cors -> cors.configurationSource(corsConfigurationSource))
            .csrf(AbstractHttpConfigurer::disable)
            .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .authorizeHttpRequests(auth -> auth
                // Allow pre-flight OPTIONS requests from browsers
                .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
                // Passenger submission & screening endpoint
                .requestMatchers(HttpMethod.POST, "/api/v1/screenings/submit-and-validate").permitAll()
                // Admin review queue
                .requestMatchers(HttpMethod.GET, "/api/v1/screenings/admin/**").permitAll()
                // Health check & Actuator for cloud monitoring
                .requestMatchers("/actuator/health", "/health").permitAll()
                // All other requests require authentication
                .anyRequest().authenticated()
            );

        return http.build();
    }
}