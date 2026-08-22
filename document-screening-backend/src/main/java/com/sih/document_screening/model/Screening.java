package com.sih.document_screening.model;

import java.time.LocalDateTime;
import java.util.UUID;

import org.hibernate.annotations.CreationTimestamp;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Entity
@Table(name = "screenings")
@Getter
@Setter
@NoArgsConstructor
public class Screening {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(nullable = false)
    private String claimedNationality;

    @Enumerated(EnumType.STRING)
    private ScreeningStatus status = ScreeningStatus.IN_PROGRESS;

    private Integer finalRiskScore; // 0 to 100
    private String riskCategory;    // LOW, MEDIUM, HIGH

    @OneToOne(mappedBy = "screening", cascade = CascadeType.ALL, orphanRemoval = true)
    private PassportRecord passportRecord;

    @OneToOne(mappedBy = "screening", cascade = CascadeType.ALL, orphanRemoval = true)
    private VisaRecord visaRecord;

    @OneToOne(mappedBy = "screening", cascade = CascadeType.ALL, orphanRemoval = true)
    private NationalIdRecord nationalIdRecord;

    @OneToOne(mappedBy = "screening", cascade = CascadeType.ALL, orphanRemoval = true)
    private VerificationScore verificationScore;

    @CreationTimestamp
    private LocalDateTime createdAt;
}